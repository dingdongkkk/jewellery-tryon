"""Image regressions: exported detail, compositing, and rejected generation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import pipeline as P
from app.tryon import run_tryon
from app.providers.base import GenerationResult


class DetailPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        y, x = np.mgrid[:500, :400]
        photo = np.stack([80 + x // 4, 90 + y // 5, 100 + (x + y) // 8], -1).astype('uint8')
        Image.fromarray(photo).save(self.root / 'person.png')
        y, x = np.mgrid[:240, :180]
        # Real high-frequency detail that disappears if a small composite is enlarged.
        texture = np.where((x // 3 + y // 3) % 2, 210, 65).astype('uint8')
        alpha = np.where((x > 5) & (x < 174) & (y > 5) & (y < 234), 255, 0).astype('uint8')
        rgba = np.stack([texture, texture, texture, alpha], -1)
        Image.fromarray(rgba).save(self.root / 'cut.png')
        self.product = dict(cutout_path=str(self.root / 'cut.png'), original_path=str(self.root / 'cut.png'), category='earring')
        self.upload = dict(original_path=str(self.root / 'person.png'))
        self.job = dict(provider='none', seed=7, placement={'items': [dict(cx=200, cy=250, width=90, rotation=19)]},
                        settings={'output_long_side': 1024, 'auto_light': False, 'shadow_opacity': 0})

    def run_job(self):
        result = run_tryon(self.job, self.product, self.upload, self.root / 'out', lambda _: None)
        return result, np.asarray(Image.open(result['output_path']))

    def test_closeup_is_literal_master_crop(self):
        result, master = self.run_job()
        q = result['info']['quality']
        self.assertEqual(list(master.shape[1::-1]), q['output_size'])
        self.assertGreater(master.shape[0], 500)
        for tile in result['info']['tiles']:
            x, y, w, h = tile['region']
            crop = np.asarray(Image.open(self.root / 'out' / tile['file']))
            np.testing.assert_array_equal(crop, master[y:y+h, x:x+w])

    def test_strips_match_single_pass_and_keep_texture(self):
        result, actual = self.run_job()
        k = result['info']['quality']['export_scale']
        base = Image.open(self.root / 'person.png').resize((actual.shape[1], actual.shape[0]), Image.Resampling.LANCZOS)
        expected = np.asarray(base, dtype=np.float32) / 255
        cut = P.premultiply(P.load_rgba(self.root / 'cut.png'))
        items = [P.Item.from_dict(it) for it in self.job['placement']['items']]
        P.render_layer(cut, items, 240/180, (0, 0, 400, 500), k, shadow_opacity=0)(expected)
        delta = np.abs(actual.astype(float) - np.asarray(P.to_pil(expected)).astype(float))
        self.assertLess(np.percentile(delta, 99.99), 2)
        # Interior retains alternating engraved detail at the larger render scale.
        self.assertGreater(actual[440:560, 350:450, 0].std(), 35)

    def test_highlight_relight_does_not_flatten_texture(self):
        canvas = np.zeros((1, 3, 3), np.float32)
        values = np.array([0.80, 0.90, 0.98], np.float32)
        rgba = np.ones((1, 3, 4), np.float32)
        rgba[0, :, :3] = values[:, None]
        P.composite_pm(canvas, rgba, 0, 0, np.full((1, 3, 3), 1.6, np.float32))
        self.assertTrue(np.all(np.diff(canvas[0, :, 0]) > 0.02))
        self.assertLess(canvas.max(), 1)

    def test_rejected_ai_cannot_replace_background_or_texture(self):
        class BadProvider:
            def generate_tryon(self, image, reference, prompt, seed, settings):
                return GenerationResult(Image.new('RGB', image.size, 'magenta'), seed)
        _, baseline = self.run_job()
        self.job['provider'] = 'qwen-fast'
        with patch('app.tryon.get_provider', return_value=BadProvider()):
            result, actual = self.run_job()
        self.assertIn('ai_rejected', result['info'])
        np.testing.assert_array_equal(actual, baseline)

    def test_invalid_output_budget_rejected(self):
        for value in (float('nan'), 0, 90000, '8192', True):
            self.job['settings']['output_long_side'] = value
            with self.assertRaises(ValueError):
                self.run_job()

    def test_each_piece_gets_large_model_crop_and_accepted_lighting(self):
        calls = []
        class LightingProvider:
            def generate_tryon(self, image, reference, prompt, seed, settings):
                calls.append(image.size)
                # A subtle exposure change with identical geometry is accepted.
                return GenerationResult(Image.fromarray((np.asarray(image) * 0.95).astype('uint8')), seed)
        self.job['provider'] = 'fal-nano-banana'
        self.job['settings']['mode'] = 'lighting'  # lighting-only pass (worn mode is tested separately)
        self.job['placement']['items'] = [dict(cx=80, cy=240, width=40), dict(cx=320, cy=240, width=40, flip=True)]
        with patch('app.tryon.get_provider', return_value=LightingProvider()):
            result, master = self.run_job()
        self.assertEqual(calls, [(1024, 1024), (1024, 1024)])
        self.assertEqual(result['info']['accepted_lighting_passes'], 2)
        self.assertTrue(all(p['jewellery_model_width'] >= 150 for p in result['info']['lighting_passes']))
        for tile in result['info']['tiles']:
            x, y, w, h = tile['region']
            np.testing.assert_array_equal(np.asarray(Image.open(self.root / 'out' / tile['file'])), master[y:y+h, x:x+w])

    def test_fractional_scale_shadows_and_chain_have_no_strip_seams(self):
        self.job['placement'] = {'items': [dict(cx=200, cy=250, width=60, rotation=23)],
                                 'chain': {'points': [[120, 100], [150, 190], [200, 220], [250, 190], [280, 100]], 'thickness': 1.8}}
        self.job['settings']['shadow_opacity'] = 0.14
        result, actual = self.run_job()
        k = 1024 / 500
        expected = np.asarray(Image.open(self.root / 'person.png').resize((actual.shape[1], actual.shape[0]), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        cut = P.load_rgba(self.root / 'cut.png')
        P.draw_chain(expected, self.job['placement']['chain'], P.dominant_metal_color(cut), (0, 0, 400, 500), k)
        P.render_layer(P.premultiply(cut), [P.Item.from_dict(self.job['placement']['items'][0])], 240/180,
                       (0, 0, 400, 500), k, shadow_opacity=0.14)(expected)
        delta = np.abs(actual.astype(float) - np.asarray(P.to_pil(expected)).astype(float))
        self.assertLess(np.percentile(delta, 99.9), 3)

    def test_mirrored_attachment_stays_at_earlobe(self):
        from app.autoplace import place_earrings
        face = np.zeros((478, 2), np.float32)
        face[234], face[454], face[1] = [100, 100], [300, 100], [200, 130]
        product = {'dimensions': {'cutout_w': 100, 'cutout_h': 120, 'approx_width_mm': 20},
                   'anchors': {'attach': [0.2, 0.1]}}
        with patch('app.autoplace._face', return_value=(face, 1, 200, 0.0)):  # front view (yaw 0)
            items = place_earrings(Image.new('RGB', (400, 500)), product)['items']
        self.assertAlmostEqual(items[0]['cx'] + (0.2 - 0.5) * 20, 92)
        self.assertAlmostEqual(items[1]['cx'] + (0.8 - 0.5) * 20, 308)

    def test_profile_gets_one_earring_on_the_visible_ear(self):
        from app.autoplace import place_earrings
        face = np.zeros((478, 2), np.float32)
        # head turned to the right: nose far right, the visible ear (234) far from it, 454 near the nose
        face[234], face[454], face[1], face[10], face[152] = [220, 260], [347, 242], [400, 247], [348, 147], [398, 352]
        product = {'dimensions': {'cutout_w': 100, 'cutout_h': 120, 'approx_width_mm': 20},
                   'anchors': {'attach': [0.5, 0.05]}}
        with patch('app.autoplace._face', return_value=(face, 1, 128, 51.0)):
            r = place_earrings(Image.new('RGB', (800, 872)), product)
        self.assertEqual(len(r['items']), 1)
        self.assertEqual(r['detected']['view'], 'profile')
        self.assertLess(r['items'][0]['cx'], 260)  # on the ear side, not on the face

    def test_occlusion_mask_matches_visible_layer(self):
        from app.tryon import _alpha_layer
        cut = P.premultiply(P.load_rgba(self.root / 'cut.png'))
        item = P.Item(200, 250, 90, occlude={'rect': [0, 0, 1, 1], 'always': True})
        alpha = _alpha_layer(cut, [item], 1, (0, 0), (500, 400))
        self.assertLess(alpha.max(), 1e-5)

    def test_no_image_response_finishes_with_local_lighting(self):
        from app.providers.base import NoImageGeneratedError
        class EmptyProvider:
            def generate_tryon(self, *args):
                raise NoImageGeneratedError('No image')
        _, baseline = self.run_job()
        self.job['provider'] = 'fal-nano-banana'
        self.job['settings']['mode'] = 'lighting'  # lighting-only pass (worn mode is tested separately)
        with patch('app.tryon.get_provider', return_value=EmptyProvider()):
            result, actual = self.run_job()
        self.assertEqual(result['info']['accepted_lighting_passes'], 0)
        self.assertTrue(result['info']['warnings'])
        np.testing.assert_array_equal(actual, baseline)


if __name__ == '__main__':
    unittest.main()
