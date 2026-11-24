from app.pipeline.decompose import parse_claims


def test_concrete_claims_kept():
    out = parse_claims('[{"text":"本市马拉松于2024年举办"},{"text":"奥运会在2020年延期"}]', 5)
    assert [c.text for c in out] == ["本市马拉松于2024年举办", "奥运会在2020年延期"]


def test_vacuous_claims_dropped():
    # 拆解器凑数用的空话（mm05 踩过：检索拉回同主题无关页面，把图文矛盾冲成"支持"）
    out = parse_claims(
        '[{"text":"某种事件将在2024年发生"},'
        '{"text":"与2024年相关的某种信息被提及"},'
        '{"text":"2024年即将到来"}]',
        5,
    )
    assert out == []


def test_degraded_claim_dropped():
    # 拆解器把「东京奥运会现场」压成「2020年发生了一件事情」：只剩数字重合，
    # VLM 对着空话判一致，PARIS 2024 的图就再也矛盾不出来了（mm01 踩过）
    out = parse_claims('[{"text":"2020年发生了一件事情"}]', 5, source_text="这张照片拍摄于2020年东京奥运会现场。")
    assert out == []


def test_claim_with_source_entities_kept():
    out = parse_claims(
        '[{"text":"照片拍摄于2020年东京奥运会现场"}]', 5, source_text="这张照片拍摄于2020年东京奥运会现场。"
    )
    assert [c.text for c in out] == ["照片拍摄于2020年东京奥运会现场"]


def test_no_source_check_keeps_everything():
    # 不给原文时不做实词重合校验，避免误杀
    out = parse_claims('[{"text":"2020年"}]', 5)
    assert len(out) == 1


def test_number_only_claim_dropped_when_source_given():
    # 只有数字、没有任何实词与原文重合的断言，是降级改写
    out = parse_claims('[{"text":"2020年"}]', 5, source_text="一些无关的话。")
    assert out == []
