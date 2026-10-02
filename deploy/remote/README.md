# Your own remote mepd web

The full app, with no demo restrictions, for your own use from anywhere
(e.g. a phone). It runs on this machine as a systemd user service, behind a
password, and a Cloudflare quick tunnel publishes it over HTTPS.

```bash
deploy/remote/run.sh password NEW   # once (also: change it; restarts it)
deploy/remote/run.sh start          # start it on the latest develop
deploy/remote/run.sh share          # prints https://<random>.trycloudflare.com
deploy/remote/run.sh status | logs | unshare | stop
```

**Anyone who logs in has your access.** They can open your files through
sessions, set compute profiles that name programs, and run jobs as you.
Use a long password and close the tunnel (`unshare`) when you don't need it.
Failed logins are throttled (10 per address per 10 minutes).

- Its workspace is `~/mepd_remote/workspace`. Other workspaces open from the
  session menu, but don't have one open here and in another `mepd web` at
  the same time: the two servers would both manage its jobs.
- It runs a clean worktree of `origin/develop` (`~/mepd_remote_src`, updated
  by every `start`), not work in progress in this checkout.
- The tunnel's URL changes every time you `share`.
