"""While GSM works on its own internal coordinates it asks for no energies,
and the live view would look frozen: the live line says so, once a minute."""

from mepd.pathminimizers.gsm import _quiet_status


def test_the_live_line_reports_a_quiet_gsm_once_a_minute():
    assert _quiet_status(30, 0) == (None, 0)                  # the first minute: nothing
    msg, reported = _quiet_status(65, 0)
    assert reported == 1 and "no new energies requested for 1 min" in msg
    assert _quiet_status(100, 1) == (None, 1)                 # same minute: no new message
    msg, reported = _quiet_status(250, 1)
    assert reported == 4 and "for 4 min" in msg
