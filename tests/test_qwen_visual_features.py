from __future__ import annotations

import unittest

import torch

from vision_memory.reader import qwen3vl_query_free_visual_features


class MockImageProcessor:
    def __init__(self, *, detach: bool = False) -> None:
        self.detach = detach
        self.observed: dict | None = None

    def __call__(self, **kwargs):
        self.observed = kwargs
        image = kwargs["images"][0]
        pixel_values = image.unsqueeze(0)
        if self.detach:
            pixel_values = pixel_values.detach()
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class MockProcessor:
    def __init__(self, *, detach: bool = False) -> None:
        self.image_processor = MockImageProcessor(detach=detach)


class MockModel(torch.nn.Module):
    def get_image_features(self, pixel_values, image_grid_thw):
        del image_grid_thw
        feature = pixel_values.flatten(2).transpose(1, 2)
        return (tuple(feature[index] for index in range(feature.shape[0])), ())


class QwenVisualFeatureTest(unittest.TestCase):
    def test_query_free_feature_preserves_image_gradient(self) -> None:
        image = torch.rand(3, 4, 4, requires_grad=True)
        processor = MockProcessor()
        output = qwen3vl_query_free_visual_features(
            model=MockModel(),
            processor=processor,
            image=image,
            device=torch.device("cpu"),
            do_resize=False,
        )
        self.assertEqual(tuple(output.features.shape), (1, 16, 3))
        self.assertEqual(set(processor.image_processor.observed), {"images", "return_tensors", "do_rescale", "do_resize"})
        output.features.square().mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertGreater(float(image.grad.norm()), 0.0)

    def test_detached_fast_path_fails_closed(self) -> None:
        image = torch.rand(3, 4, 4, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "detached"):
            qwen3vl_query_free_visual_features(
                model=MockModel(),
                processor=MockProcessor(detach=True),
                image=image,
                device=torch.device("cpu"),
            )

    def test_api_has_no_query_or_choice_surface(self) -> None:
        with self.assertRaises(TypeError):
            qwen3vl_query_free_visual_features(  # type: ignore[call-arg]
                model=MockModel(),
                processor=MockProcessor(),
                image=torch.rand(3, 4, 4),
                device=torch.device("cpu"),
                query="forbidden",
            )


if __name__ == "__main__":
    unittest.main()
