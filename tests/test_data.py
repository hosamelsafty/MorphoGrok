"""Correctness tests for the data layer — the difficulty ladder is scientifically
load-bearing (it is the grokking 'cleanliness' axis), so pin it with known roots."""
from grokkingroots.data.arabic_text import normalize_surface, strip_diacritics
from grokkingroots.data.difficulty import classify_root_type


def diff(root):
    return classify_root_type(root)["difficulty"]


def rtype(root):
    return classify_root_type(root)["root_type"]


def test_sound_triliteral_is_L1():
    assert (diff("ك ت ب"), rtype("ك ت ب")) == ("L1", "sound")
    assert (diff("د ر س"), rtype("د ر س")) == ("L1", "sound")


def test_hollow_is_L2():
    assert diff("ق و ل") == "L2" and rtype("ق و ل") == "hollow"   # q-w-l
    assert diff("ب ي ع") == "L2" and rtype("ب ي ع") == "hollow"   # b-y-ʿ


def test_defective_and_assimilated_are_L2():
    assert diff("د ع و") == "L2" and rtype("د ع و") == "defective"  # d-ʿ-w
    assert diff("و ص ل") == "L2" and rtype("و ص ل") == "assimilated"  # w-ṣ-l


def test_hamzated_is_L2():
    assert diff("أ ك ل") == "L2" and rtype("أ ك ل") == "hamzated"  # ʾ-k-l


def test_doubled_is_L3():
    assert diff("م د د") == "L3" and rtype("م د د") == "doubled"


def test_quadriliteral_is_L4():
    assert diff("د ح ر ج") == "L4" and rtype("د ح ر ج") == "quadriliteral"


def test_strip_and_normalize():
    assert strip_diacritics("كَتَبَ") == "كتب"
    assert normalize_surface("مَكْتُوبـ") == "مكتوب"  # diacritics + trailing tatweel removed
