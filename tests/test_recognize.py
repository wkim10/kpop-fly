import pytest

from kpop_fly.recognize import AudDError, Recognition, parse, parse_timecode, saved_dance

SONGS = {"fancy": {"artist": "TWICE", "title": "FANCY"}, "cheer-up": {"artist": "TWICE", "title": "CHEER UP"}}


def test_parse_success():
    rec = parse({"status": "success", "result": {
        "artist": "TWICE", "title": "TT", "timecode": "01:23", "song_link": "https://lis.tn/x",
        "apple_music": {"genreNames": ["K-Pop", "Music"]}}})
    assert (rec.artist, rec.title, rec.timecode) == ("TWICE", "TT", 83.0)
    assert rec.is_kpop and rec.name == "TWICE - TT"


def test_parse_not_found_and_errors():
    assert parse({"status": "success", "result": None}) is None
    with pytest.raises(AudDError, match="#900"):
        parse({"status": "error", "error": {"error_code": 900, "error_message": "Invalid API token"}})


@pytest.mark.parametrize("text,seconds", [("00:56", 56), ("1:02:03", 3723), ("", None), (None, None), ("x", None)])
def test_timecodes(text, seconds):
    assert parse_timecode(text) == seconds


def test_genres_without_apple_music_are_not_kpop():
    assert not Recognition("BTS", "Dynamite", 10.0).is_kpop
    assert Recognition("BTS", "Dynamite", 10.0, ["Pop"]).is_kpop is False


@pytest.mark.parametrize("artist,title,slug", [
    ("TWICE", "FANCY", "fancy"),
    ("Twice", "Cheer Up", "cheer-up"),                # case and spacing don't matter
    ("TWICE & Friends", "FANCY", "fancy"),            # extra credited artists are fine
    ("TWICE", "FANCY (Japanese ver.)", None),         # a different version is not the saved dance
    ("ITZY", "FANCY", None),                          # same title, different artist
    ("TWICE", "LIKEY", None),
])
def test_saved_dance(artist, title, slug):
    assert saved_dance(Recognition(artist, title, 0.0), SONGS) == slug
