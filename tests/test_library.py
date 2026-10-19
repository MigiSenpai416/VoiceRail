import zipfile

import pytest
import torch

from voicerail import library, paths
from voicerail.rvc.checkpoint import ModelFormatError, export_voice_model
from voicerail.rvc.nn import Synthesizer
from voicerail.train.config import AUDIO_CONFIGS


@pytest.fixture
def model_file(tmp_path):
    config = AUDIO_CONFIGS[40000].synthesizer()
    return export_voice_model(
        Synthesizer(config).state_dict(),
        config,
        tmp_path / "Voice.pth",
        name="Voice",
        epoch=5,
        step=1,
    )


def test_import_pth_and_list(model_file):
    info = library.import_files([model_file])
    assert info.name == "Voice" and info.epochs == 5 and not info.has_index
    assert [m.name for m in library.list_models()] == ["Voice"]
    again = library.import_files([model_file])
    assert again.name == "Voice (2)"


def test_import_zip_ignores_paths_and_other_files(model_file, tmp_path):
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(model_file, "../../evil/Voice.pth")
        zf.writestr("readme.txt", "hello")
        zf.writestr("G_2333333.pth", b"junk")
    info = library.import_files([archive], "Packed")
    folder = paths.models_dir() / info.name
    assert sorted(p.name for p in folder.iterdir()) == ["Packed.pth", "info.json"]
    assert not (tmp_path.parent / "evil").exists()


def test_import_rejects_non_models(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"\x00" * 10)
    with pytest.raises(ModelFormatError):
        library.import_files([bad])
    assert library.list_models() == []


def test_rename_and_delete(model_file):
    library.import_files([model_file], "First")
    new = library.rename_model("First", "Second")
    model, index = library.model_paths(new)
    assert model.name == "Second.pth" and index is None
    library.delete_model(new)
    assert library.list_models() == []


def test_safe_name():
    assert paths.safe_name("  my/voice:*?  ") == "my_voice"
    assert paths.safe_name("...") == "untitled"
    assert paths.safe_name("Song (Live) [2024]") == "Song (Live) [2024]"


def test_weights_only_loading_blocks_pickles(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    path = tmp_path / "evil.pth"
    torch.save({"weight": Evil(), "config": []}, path)
    with pytest.raises(ModelFormatError):
        library.import_files([path])


def test_safe_name_is_idempotent():
    name = "a" * 79 + " b"
    once = paths.safe_name(name)
    assert paths.safe_name(once) == once and not once.endswith(" ")


def test_model_names_cannot_escape_library(model_file):
    library.import_files([model_file], "Ok")
    with pytest.raises(FileNotFoundError):
        library.model_paths("../training")
    with pytest.raises(FileNotFoundError):
        library.delete_model("..")


def test_rejects_foreign_embedder(tmp_path):
    config = AUDIO_CONFIGS[40000].synthesizer()
    path = export_voice_model(
        Synthesizer(config).state_dict(),
        config,
        tmp_path / "Spin.pth",
        name="Spin",
        epoch=1,
        step=1,
    )
    data = torch.load(path, weights_only=True)
    data["embedder_model"] = "spin"
    torch.save(data, path)
    with pytest.raises(ModelFormatError, match="embedder"):
        library.import_files([path])
