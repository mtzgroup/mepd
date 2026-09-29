"""What the web UI shows as a running job's live line: the last meaningful
line (not a table row a CLI printed), and CREST MSReact's progress while
it runs (it would otherwise be silent for minutes)."""

import os
import textwrap

from mepd.web.jobs import _status_line


def test_the_live_line_skips_tables_and_the_progress_prefix():
    lines = ["Expanding the network from the seed: 1 round(s), products proposed by crest-msreact.",
             "\x1b[36m│ positive_steps_thre │ 15     │\x1b[0m", "╰─────────────────────┴────────╯", ""]
    assert _status_line(lines).startswith("Expanding the network")
    assert _status_line(["[main] CREST MSReact: 12/32 distorted structures optimized"]) == \
        "CREST MSReact: 12/32 distorted structures optimized"
    assert _status_line(["│ a │ b │", ""]) == ""


def test_crest_msreact_progress_reaches_the_status_line(tmp_path, monkeypatch):
    """A stand-in `crest` printing msreact's progress the way CREST does
    ('# of distortions N', then 1 2 3 ... as each is optimized)."""
    import mepd.discovery.crest_msreact as cm
    import mepd.progress as progress

    fake = tmp_path / "crest"
    fake.write_text(textwrap.dedent("""\
        #!/bin/sh
        echo "  # of distortions          4"
        for k in 1 2 3 4; do printf " $k"; sleep 0.6; done
        echo
        echo " done."
        """))
    fake.chmod(0o755)
    seen = []
    monkeypatch.setattr(progress, "update_status", lambda message: seen.append(message))
    code, log = cm._run_with_progress([str(fake)], tmp_path, dict(os.environ), timeout=30)
    assert code == 0 and "done." in log
    assert seen[0].startswith("CREST MSReact: distorting")
    counts = [m for m in seen if "/4 distorted structures optimized" in m]
    assert counts and counts[-1].startswith("CREST MSReact: 4/4")
    assert len(counts) >= 2          # reported while it ran, not only at the end
