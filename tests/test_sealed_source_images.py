import pytest
from PIL import Image

from vision_memory.dreamlite.source_images import load_sealed_source_image
from vision_memory.training.latent_bank_unet import file_sha256


def test_sealed_source_reads_exact_pixels_and_rejects_changed_bytes(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), (12, 34, 56)).save(path)
    group = {"source_kind": "sealed_rgb_1024", "source_image_path": "source.png",
             "source_image_file_sha256": file_sha256(path)}
    image, resolved = load_sealed_source_image(group, tmp_path / "manifest.json")
    assert resolved == path and image.getpixel((731, 829)) == (12, 34, 56)
    Image.new("RGB", (1024, 1024), (128, 128, 128)).save(path)
    with pytest.raises(ValueError, match="file changed"):
        load_sealed_source_image(group, tmp_path / "manifest.json")


@pytest.mark.parametrize("mode,size", [("L", (1024, 1024)), ("RGB", (512, 512))])
def test_source_contract_does_not_silently_convert_or_resize(tmp_path, mode, size):
    path = tmp_path / "source.png"
    Image.new(mode, size).save(path)
    group = {"source_kind": "sealed_rgb_1024", "source_image_path": str(path),
             "source_image_file_sha256": file_sha256(path)}
    with pytest.raises(ValueError, match="RGB 1024x1024"):
        load_sealed_source_image(group, tmp_path / "manifest.json")
