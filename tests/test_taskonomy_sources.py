from functools import partial
import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFile

from omnitaskonomy.data.common import read_jsonl, resolve_image, sha256
from omnitaskonomy.data import taskonomy


class ArchiveHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append(self.path)
        super().do_GET()

    def log_message(self, format, *args):
        pass


class TaskonomySourcesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.hub = self.root / "hub"
        self.hub.mkdir()
        self.archives = self.root / "archives"
        self.archives.mkdir()
        self.raw = self.root / "raw"
        self.output = self.root / "prepared/train.jsonl"
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(ArchiveHandler, directory=str(self.archives)))
        self.server.requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.ids = ["allensville/point_9_view_0", "allensville/point_2_view_1",
                    "allensville/point_9_view_0"]
        self.inputs = [Image.new("RGB", (512, 512), color)
                       for color in ((25, 90, 155), (200, 110, 20))]
        self.targets = [Image.new("RGB", (512, 512), color)
                        for color in ((30, 50, 70), (180, 160, 140))]
        self.task = "normal"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @staticmethod
    def ordered_hash(values):
        return hashlib.sha256("".join(value + "\n" for value in values).encode()).hexdigest()

    def image_hash(self, images):
        by_id = dict(zip(dict.fromkeys(self.ids), images))
        return self.ordered_hash([
            hashlib.sha256(by_id[uid].convert("RGB").tobytes()).hexdigest()
            for uid in self.ids
        ])

    def write_archive(self, domain, images, *, member_kind=None):
        filename = f"allensville_{domain}.tar"
        path = self.archives / filename
        with tarfile.open(path, "w") as archive:
            for uid, image in zip(dict.fromkeys(self.ids), images):
                point_view = uid.split("/")[1]
                member = tarfile.TarInfo(f"{domain}/{point_view}_domain_{domain}.png")
                if member_kind == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = str(self.root / "outside.png")
                    archive.addfile(member)
                elif member_kind == "traversal":
                    member.name = f"../{member.name}"
                    payload = io.BytesIO()
                    image.save(payload, format="PNG")
                    member.size = payload.tell()
                    payload.seek(0)
                    archive.addfile(member, payload)
                else:
                    payload = io.BytesIO()
                    image.save(payload, format="PNG")
                    member.size = payload.tell()
                    payload.seek(0)
                    archive.addfile(member, payload)
        return {"filename": filename, "url": f"{self.base_url}/{filename}",
                "md5": hashlib.md5(path.read_bytes()).hexdigest()}

    def configure(self, *, task="normal", target_domain="normal", target_transform="rgb",
                  input_transform="rgb", prepared_inputs=None, prepared_targets=None,
                  member_kind=None):
        self.task = task
        name = taskonomy.TASKS[task].split(":")[1]
        self.selection = self.hub / f"sources/taskonomy/selections/{name}.jsonl.gz"
        self.selection.parent.mkdir(parents=True, exist_ok=True)
        self.selection.write_bytes(gzip.compress(
            "".join(json.dumps({"id": uid}) + "\n" for uid in self.ids).encode(), mtime=0))
        archives = [self.write_archive("rgb", self.inputs)]
        if target_domain != "rgb":
            archives.append(self.write_archive(target_domain, self.targets, member_kind=member_kind))
        self.spec = {
            "task_id": taskonomy.TASKS[task], "task_name": task, "r": "Reconstruction",
            "modality": "i2i", "usage": "train", "rows": len(self.ids),
            "unique_ids": len(set(self.ids)), "repeated_rows": len(self.ids) - len(set(self.ids)),
            "selection_file": self.selection.relative_to(self.hub).as_posix(),
            "compressed_bytes": self.selection.stat().st_size,
            "compressed_sha256": sha256(self.selection),
            "ordered_id_sha256": self.ordered_hash(self.ids),
            "editing_prompt": "Produce the target image.",
            "selection_provenance": "Local fixture in saved occurrence order.",
            "input_domain": "rgb", "target_domain": target_domain,
            "input_transform": input_transform, "target_transform": target_transform,
            "buildings": ["allensville"],
            "archives": [archive["filename"] for archive in archives],
            "ordered_input_rgb_sha256": self.image_hash(
                self.inputs if prepared_inputs is None else prepared_inputs),
            "ordered_output_rgb_sha256": self.image_hash(
                self.targets if prepared_targets is None else prepared_targets),
            "decode_recoveries": [],
        }
        self.catalog = {
            "schema_version": 1, "source_dataset": "Taskonomy", "modality": "i2i",
            "usage": "train", "task_count": 1, "total_rows": len(self.ids),
            "images_included": False, "validation_included": False,
            "extracted_path_template": "{domain}/taskonomy/{building}/{point_view}_domain_{domain}.png",
            "tasks": [self.spec], "archives": archives,
        }
        self.save_catalog()

    def save_catalog(self):
        (self.hub / "sources/taskonomy/sources.json").write_text(json.dumps(self.catalog))

    def prepare(self):
        def hub_download(repo_id, filename, **kwargs):
            return str(self.hub / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=hub_download):
            return taskonomy.prepare_taskonomy(
                self.task, self.output, raw_root=self.raw, revision="f" * 40, workers=1)

    def assert_manifest_pixels(self, inputs, targets):
        rows = list(read_jsonl(self.output))
        self.assertEqual([row["source_uid"] for row in rows], self.ids)
        self.assertEqual(len({row["uid"] for row in rows}), len(self.ids))
        expected_inputs = dict(zip(dict.fromkeys(self.ids), inputs))
        expected_targets = dict(zip(dict.fromkeys(self.ids), targets))
        for uid, row in zip(self.ids, rows):
            self.assertEqual(row["prompt"], self.spec["editing_prompt"])
            for key, expected in (("source_image", expected_inputs), ("target_image", expected_targets)):
                with Image.open(resolve_image(self.output, row[key])) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.tobytes(), expected[uid].tobytes())

    def test_download_preserves_order_repeats_pixels_and_cached_data(self):
        self.configure()
        self.assertEqual(self.prepare(), self.output)
        self.assert_manifest_pixels(self.inputs, self.targets)
        before = self.output.read_bytes()
        requests = list(self.server.requests)
        self.prepare()
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.server.requests, requests)

    def test_colorization_uses_original_rgb_as_target(self):
        grayscale = [image.convert("L").convert("RGB") for image in self.inputs]
        self.configure(task="colorization", target_domain="rgb", input_transform="rgb_to_gray",
                       prepared_inputs=grayscale, prepared_targets=self.inputs)
        self.prepare()
        self.assert_manifest_pixels(grayscale, self.inputs)
        self.assertEqual(self.server.requests, ["/allensville_rgb.tar"])

    def test_missing_rendered_image_is_repaired_without_downloading(self):
        grayscale = [image.convert("L").convert("RGB") for image in self.inputs]
        self.configure(task="colorization", target_domain="rgb", input_transform="rgb_to_gray",
                       prepared_inputs=grayscale, prepared_targets=self.inputs)
        self.prepare()
        before = self.output.read_bytes()
        requests = list(self.server.requests)
        first = next(read_jsonl(self.output))
        missing = resolve_image(self.output, first["source_image"])
        missing.unlink()
        self.prepare()
        self.assertTrue(missing.is_file())
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.server.requests, requests)
        self.assert_manifest_pixels(grayscale, self.inputs)

    def test_metadata_write_failure_leaves_manifest_unpublished_and_can_retry(self):
        self.configure()
        write_text = Path.write_text

        def fail_metadata(path, *args, **kwargs):
            if path.name == "train.preparing.metadata.json":
                raise OSError("No space for metadata")
            return write_text(path, *args, **kwargs)

        with patch.object(Path, "write_text", new=fail_metadata):
            with self.assertRaisesRegex(OSError, "No space for metadata"):
                self.prepare()
        self.assertFalse(self.output.exists())
        requests = list(self.server.requests)
        self.prepare()
        self.assert_manifest_pixels(self.inputs, self.targets)
        self.assertEqual(self.server.requests, requests)
        metadata = json.loads(self.output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["manifest_sha256"], sha256(self.output))

    def test_depth_masks_invalid_pixels_and_handles_constant_images(self):
        depth = np.tile(np.array([100, 200, 300, 65535], dtype=np.uint16), (512, 128))
        self.targets = [Image.fromarray(depth), Image.new("I;16", (512, 512), 65535)]
        expected = [Image.fromarray(np.tile(np.array([0, 127, 255, 255], dtype=np.uint8),
                                           (512, 128))).convert("RGB"),
                    Image.new("RGB", (512, 512))]
        self.configure(task="depth_zbuffer", target_domain="depth_zbuffer", target_transform="depth16",
                       prepared_targets=expected)
        self.prepare()
        self.assert_manifest_pixels(self.inputs, expected)

    def test_minmax_renders_full_16_bit_range_and_constant_images(self):
        values = np.tile(np.array([0, 32768, 65535, 0], dtype=np.uint16), (512, 128))
        self.targets = [Image.fromarray(values), Image.new("I;16", (512, 512), 300)]
        expected = [Image.fromarray(np.tile(np.array([0, 127, 255, 0], dtype=np.uint8),
                                           (512, 128))).convert("RGB"),
                    Image.new("RGB", (512, 512))]
        self.configure(task="edge_texture", target_domain="edge_texture", target_transform="minmax16",
                       prepared_targets=expected)
        self.prepare()
        self.assert_manifest_pixels(self.inputs, expected)

    def test_selection_corruption_fails_before_archive_download(self):
        self.configure()
        self.selection.write_bytes(self.selection.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.server.requests, [])
        self.assertFalse(self.output.exists())

    def test_selection_order_mismatch_fails_before_archive_download(self):
        self.configure()
        reordered = [self.ids[0], self.ids[2], self.ids[1]]
        self.selection.write_bytes(gzip.compress(
            "".join(json.dumps({"id": uid}) + "\n" for uid in reordered).encode(), mtime=0))
        self.spec["compressed_sha256"] = sha256(self.selection)
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            self.prepare()
        self.assertEqual(self.server.requests, [])
        self.assertFalse(self.output.exists())

    def test_pixel_mismatch_leaves_no_completed_manifest(self):
        self.configure()
        self.spec["ordered_output_rgb_sha256"] = "0" * 64
        self.save_catalog()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_archive_checksum_mismatch_leaves_no_completed_manifest(self):
        self.configure()
        self.catalog["archives"][1]["md5"] = "0" * 32
        self.save_catalog()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_selected_symlink_is_not_followed(self):
        self.configure(member_kind="symlink")
        self.targets[0].save(self.root / "outside.png")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_traversal_member_cannot_supply_selected_image(self):
        self.configure(member_kind="traversal")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "normal").exists())


class TaskonomyRenderingTests(unittest.TestCase):
    def test_transparency_is_composited_over_white(self):
        rgba = Image.new("RGBA", (2, 1))
        rgba.putdata([(200, 30, 10, 128), (20, 40, 60, 0)])
        self.assertEqual(list(taskonomy.render_image(rgba, "rgb").getdata()),
                         [(227, 142, 132), (255, 255, 255)])
        palette = Image.new("P", (2, 1))
        palette.putpalette([20, 30, 40, 70, 80, 90])
        palette.putdata([0, 1])
        palette.info["transparency"] = 0
        self.assertEqual(list(taskonomy.render_image(palette, "rgb").getdata()),
                         [(255, 255, 255), (70, 80, 90)])

    def test_gray_replicates_luminance_and_rejects_unknown_transform(self):
        image = Image.new("RGB", (1, 1), (10, 20, 30))
        self.assertEqual(taskonomy.render_image(image, "gray").getpixel((0, 0)), (18, 18, 18))
        with self.assertRaisesRegex(ValueError, "Unknown Taskonomy transform"):
            taskonomy.render_image(image, "percentile")

    def test_depth_uses_all_pixels_when_fewer_than_sixteen_are_valid(self):
        image = Image.fromarray(np.array([[100, 200], [300, 65535]], dtype=np.uint16))
        self.assertEqual(list(taskonomy.render_image(image, "depth16").getdata()),
                         [(0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 255, 255)])

    def test_damaged_image_requires_recorded_source_and_pixel_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            destination = root / "prepared.png"
            noise = np.random.RandomState(42).randint(0, 256, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(noise).save(source, compress_level=0)
            source.write_bytes(source.read_bytes()[:source.stat().st_size // 2])
            original_flag = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
                with self.assertRaises(OSError):
                    taskonomy._prepare_image(source, "rgb", destination, None)
                self.assertFalse(destination.exists())
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(source) as image:
                    pixels = image.convert("RGB").tobytes()
                ImageFile.LOAD_TRUNCATED_IMAGES = False
                recovery = {"source_file_sha256": sha256(source),
                            "decoded_rgb_sha256": hashlib.sha256(pixels).hexdigest()}
                path, digest = taskonomy._prepare_image(source, "rgb", destination, recovery)
                self.assertEqual(path, str(destination))
                self.assertEqual(digest, recovery["decoded_rgb_sha256"])
                with Image.open(destination) as image:
                    self.assertEqual(image.tobytes(), pixels)
                self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)
                for field in ("source_file_sha256", "decoded_rgb_sha256"):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        taskonomy._prepare_image(source, "rgb", root / "invalid.png",
                                                 dict(recovery, **{field: "0" * 64}))
                    self.assertFalse((root / "invalid.png").exists())
                    self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = original_flag


if __name__ == "__main__":
    unittest.main()
