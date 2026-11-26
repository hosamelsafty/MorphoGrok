from grokkingroots.data.arabic_text import lenient_key, lenient_match


def test_exact_match_is_lenient_match():
    assert lenient_match("كتب", "كتب")


def test_waw_ya_swap_in_any_radical_position():
    assert lenient_match("قيل", "قول")      # middle radical
    assert lenient_match("رمو", "رمي")      # final radical
    assert lenient_match("يسر", "وسر")      # first radical


def test_nothing_else_is_relaxed():
    assert not lenient_match("قال", "قول")  # alif is not waw
    assert not lenient_match("رمى", "رمي")  # alif-maqsura is not ya
    assert not lenient_match("أكل", "وكل")  # hamza is not waw
    assert not lenient_match("قو", "قول")   # length must agree
    assert not lenient_match("كتب", "كتم")


def test_key_is_idempotent_and_merges_only_the_pair():
    assert lenient_key("طوي") == lenient_key("طيي") == "طوو"
    assert lenient_key(lenient_key("بيع")) == lenient_key("بيع")
