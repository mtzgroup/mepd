# Public mepd demo

A password-protected copy of `mepd web` that friends can use from any
browser, including a phone. It runs on this machine inside a locked-down
container, and a Cloudflare quick tunnel publishes it over HTTPS.

```bash
deploy/demo/run.sh start      # start the demo container (private); prints the password
deploy/demo/run.sh share      # make it public: prints https://<random>.trycloudflare.com
deploy/demo/run.sh status     # is it running, is it public, and at which URL
deploy/demo/run.sh unshare    # close the public link (the demo keeps running, private)
deploy/demo/run.sh stop       # close the link and stop the demo
deploy/demo/run.sh logs       # follow the server log
```

`share` uses a Cloudflare quick tunnel (`cloudflared`, no account): the hostname is random and
**changes every time you `share`**, so resend the link after a restart. Tailscale Funnel would give
a stable `https://<machine>.<tailnet>.ts.net` instead, but on an organization tailnet it needs an
admin to allow Funnel for the machine.

Set or change the shared password with `deploy/demo/run.sh password NEW`. It is saved in
`~/mepd_demo/root/.demo_password` and the demo restarts to use it; logged-in visitors stay logged in.
(`MEPD_DEMO_PASSWORD=... deploy/demo/run.sh start` also works; it is written to the same file.)
Without either, a random password is generated once.

## What visitors get

- **Their own workspace.** Every browser that logs in is a separate visitor. Their workspaces live in `~/mepd_demo/root/visitors/<id>/`, and no visitor can see another's work.
- **The full app, minus admin powers.** They can add structures, build the graph, run calculations (with the live view) and download results.
- **Your compute profiles, read-only.** They use the profiles in `~/mepd_demo/root/profiles/`, starting with `default` (g-xTB NEB) and `gsm` (g-xTB GSM). Edit those files on disk to change what visitors run with.
- **Limits** (`mepd/web/demo.py`, `DemoPolicy`):

  | Limit | Value |
  |---|---|
  | Atoms per structure | 40 |
  | Structures per session | 60 |
  | Sessions per visitor | 5 |
  | Calculations at a time, per visitor | 2 |
  | Calculations running, across everyone | 4 |
  | Run time per calculation | 30 min |
  | Heavy parameters | capped (channels workers, basin-hopping rounds, …) |

## Why it is safe to expose

Four layers stack:

1. **Server-side demo mode (`mepd web --demo`).**
   - No endpoint accepts a filesystem path (no "open existing output", no opening sessions by path).
   - Profiles cannot be written or validated. A profile can name executables and read files, so this matters.
   - Every job runs in the visitor's own workspace.
   - The limits above are enforced on the server, not only in the page.
2. **Login.**
   - One shared password, compared in constant time.
   - Failed logins are throttled per client and globally.
   - Each visitor gets a signed, HttpOnly cookie. The signing key is `~/mepd_demo/root/.demo_secret`; delete it to log everyone out.
3. **The container (`run.sh`).**
   - Runs as your uid, not root, with no Linux capabilities and no privilege escalation.
   - Read-only root filesystem, and capped CPU, memory and process count.
   - It sees only the `mepd/` package, its `.venv`, the Python interpreter and the g-xTB, CREST and GSM installs, all read-only, plus `~/mepd_demo` read-write. Nothing else from your home directory is visible: no `.ssh`, no other workspaces, no ChemCloud credentials.
   - The port is published on `127.0.0.1` only.
4. **The tunnel.** Cloudflare terminates HTTPS and forwards only to that loopback port. Nothing listens on a public interface of this machine.

Anyone with the password can use your CPU within those limits. Share the password only with people
you'd lend compute to, and `unshare` when you're not demoing.
