(function () {
  var data = window.OLYMPUS_CONSOLE;
  var pub = window.OLYMPUS_PUBLIC;
  var nav = document.getElementById("console-nav");
  var panel = document.getElementById("console-panel");
  var live = document.getElementById("console-live");
  var modal = document.getElementById("private-modal");
  var modalBody = document.getElementById("private-modal-body");
  var modalClose = document.getElementById("private-modal-close");
  var lastFocus = null;

  if (!data || !nav || !panel) return;

  function announce(text) {
    if (live) live.textContent = text;
  }

  function openPrivateModal(name) {
    lastFocus = document.activeElement;
    modalBody.textContent =
      (name ? name + " — " : "") + data.empty.privateRepo;
    modal.hidden = false;
    modalClose.focus();
  }

  function closePrivateModal() {
    modal.hidden = true;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function renderGlanceDiagram() {
    var cfg = data.configuration;
    var del = data.delivery;
    var inf = data.infrastructure;
    var rel =
      '<ul class="rel-row">' +
      '<li class="rel-node"><h2>Configuration</h2><p class="mono">' +
      cfg.ssotFile +
      '</p><p class="mono">' +
      cfg.modelId +
      "</p><p>" +
      cfg.ssotRepo +
      "</p></li>" +
      '<li class="rel-node"><h2>Delivery</h2><p>' +
      del.cicd +
      "</p><p>" +
      del.infraWorkflows +
      "</p></li>" +
      '<li class="rel-node"><h2>Infrastructure</h2><p class="mono">' +
      inf.cluster +
      '</p><p class="mono">' +
      inf.region +
      "</p><p>Active branch <span class=\"mono\">" +
      inf.activeBranch +
      "</span></p><p>Parked branch <span class=\"mono\">" +
      inf.parkedBranch +
      "</span></p></li></ul>";
    var services = data.topology.services
      .map(function (svc) {
        var edge = svc.public
          ? '<a href="' + svc.url + '">' + svc.url.replace("https://", "") + "</a>"
          : "ClusterIP";
        return (
          '<li class="topo-node"><p class="topo-name mono">' +
          svc.name +
          '</p><p class="topo-role">' +
          svc.role +
          '</p><p class="topo-edge">' +
          edge +
          "</p></li>"
        );
      })
      .join("");
    return (
      '<figure class="glance-diagram">' +
      '<figcaption class="visually-hidden">Configuration, Delivery, Infrastructure, and platform services</figcaption>' +
      rel +
      '<ul class="topo-services">' +
      services +
      "</ul></figure>"
    );
  }

  function renderTopology() {

    var priv = (pub.privateRepos || [])
      .map(function (repo) {
        return (
          '<button type="button" class="btn js-private" data-repo="' +
          repo.name +
          '">' +
          repo.name +
          "</button>"
        );
      })
      .join(" ");

    return (
      "<h1>Platform Topology</h1>" +
      "<p class=\"lede\">" +
      data.topology.note +
      "</p>" +
      '<div class="alert-calm" role="status">' +
      '<p class="alert-title">' +
      data.empty.n8nOfflineTitle +
      "</p>" +
      "<p>" +
      data.empty.n8nOfflineBody +
      "</p>" +
      '<p class="alert-actions">' +
      '<a href="https://n8n.levkesha.com">Retry n8n.levkesha.com</a>' +
      "<span aria-hidden=\"true\">·</span>" +
      '<a href="architecture.html">Architecture</a>' +
      "</p></div>" +
      renderGlanceDiagram() +
      "<h2>Private evidence</h2>" +
      "<p class=\"muted\">No GitHub links. Open a name for the private-repo note.</p>" +
      "<p>" +
      priv +
      "</p>"
    );
  }

  function renderConfiguration() {
    var cfg = data.configuration;
    return (
      "<h1>Configuration</h1>" +
      "<article class=\"card\"><dl class=\"kvs\">" +
      "<dt>SSOT file</dt><dd>" +
      cfg.ssotFile +
      "</dd>" +
      "<dt>SSOT repo</dt><dd>" +
      cfg.ssotRepo +
      "</dd>" +
      "<dt>Model ID</dt><dd>" +
      cfg.modelId +
      "</dd>" +
      "</dl></article>"
    );
  }

  function renderDelivery() {
    var d = data.delivery;
    return (
      "<h1>Delivery</h1>" +
      "<article class=\"card\"><p>" +
      d.cicd +
      "</p><p>" +
      d.infraWorkflows +
      "</p></article>"
    );
  }

  function renderInfrastructure() {
    var i = data.infrastructure;
    return (
      "<h1>Infrastructure</h1>" +
      "<article class=\"card\"><dl class=\"kvs\">" +
      "<dt>IaC</dt><dd>" +
      i.iac +
      "</dd>" +
      "<dt>Cluster</dt><dd>" +
      i.cluster +
      "</dd>" +
      "<dt>Region</dt><dd>" +
      i.region +
      "</dd>" +
      "<dt>Active branch</dt><dd>" +
      i.activeBranch +
      "</dd>" +
      "<dt>Parked branch</dt><dd>" +
      i.parkedBranch +
      "</dd>" +
      "</dl></article>" +
      "<p><button type=\"button\" class=\"btn js-private\" data-repo=\"infrastructure\">Why no GitHub link?</button></p>"
    );
  }

  function metricTile(title, value) {
    return (
      '<article class="metric-tile"><h2>' +
      title +
      "</h2><p>" +
      value +
      "</p></article>"
    );
  }

  function renderSpend() {
    var s = data.spend;
    var tiles = s.surfaces
      .map(function (row) {
        return metricTile(row.name, row.access);
      })
      .join("");
    var hops = s.hops
      .map(function (h) {
        return metricTile(h.name, h.path);
      })
      .join("");
    return (
      "<h1>Services &amp; Spend</h1>" +
      '<p class="lede">Spend · fixture glance</p>' +
      '<div class="metric-tiles">' +
      tiles +
      "</div>" +
      '<div class="metric-tiles">' +
      hops +
      "</div>"
    );
  }

  function renderCvJobs() {
    var cj = data.cvjobs;
    return (
      "<h1>" +
      cj.title +
      "</h1>" +
      "<p>" +
      cj.lede +
      "</p>" +
      '<p><span class="badge">' +
      cj.chip +
      "</span></p>" +
      "<p><a class=\"btn btn-primary\" href=\"" +
      cj.demoUrl +
      "\">Open CV×Jobs Demo</a></p>" +
      "<p class=\"mono muted\">" +
      cj.demoUrl +
      "</p>"
    );
  }

  function renderHeadroom() {
    var hr = data.headroom;
    return (
      "<h1>" +
      hr.title +
      "</h1>" +
      "<p><a class=\"btn btn-primary\" href=\"" +
      hr.adminUrl +
      "\">Open Headroom Admin</a></p>" +
      "<p class=\"mono muted\">" +
      hr.adminUrl +
      "</p>" +
      "<h2>Secondary n8n demo</h2>" +
      "<p class=\"mono muted\">POST " +
      hr.webhookUrl +
      "</p>" +
      '<p><button type="button" class="btn js-headroom-run">Run compress probe</button></p>' +
      '<div id="headroom-result" class="headroom-result" aria-live="polite"></div>'
    );
  }

  function renderN8n() {
    var n = data.n8n;
    var flows = (n.workflows || [])
      .map(function (w) {
        return (
          '<li><span class="mono">' +
          w.name +
          "</span> — " +
          w.note +
          "</li>"
        );
      })
      .join("");
    return (
      "<h1>" +
      n.title +
      "</h1>" +
      "<p>" +
      n.sell +
      "</p>" +
      '<p class="muted">' +
      n.demo +
      "</p>" +
      '<p><a class="btn btn-primary" href="' +
      n.url +
      '">Open n8n</a></p>' +
      '<p class="mono muted">' +
      n.url +
      "</p>" +
      (flows ? '<ul class="muted">' + flows + "</ul>" : "")
    );
  }

  function renderLiteLLM() {
    var lt = data.litellm;
    return (
      "<h1>" +
      lt.title +
      "</h1>" +
      "<p><a class=\"btn btn-primary\" href=\"" +
      lt.adminUrl +
      "\">Open LiteLLM Admin UI</a></p>" +
      "<p class=\"mono muted\">" +
      lt.adminUrl +
      "</p>" +
      "<h2>Break-glass (port-forward)</h2>" +
      "<p class=\"mono muted\">" +
      lt.screenshare.command +
      "</p>" +
      "<p class=\"mono muted\">" +
      lt.screenshare.localUrl +
      "</p>" +
      "<h2>Secondary health probe</h2>" +
      "<p class=\"mono muted\">POST " +
      lt.webhookUrl +
      "</p>" +
      '<p><button type="button" class="btn js-litellm-run">Run health probe</button></p>' +
      '<div id="litellm-result" class="headroom-result" aria-live="polite"></div>'
    );
  }

  var renderers = {
    topology: renderTopology,
    configuration: renderConfiguration,
    delivery: renderDelivery,
    infrastructure: renderInfrastructure,
    spend: renderSpend,
    cvjobs: renderCvJobs,
    headroom: renderHeadroom,
    litellm: renderLiteLLM,
    n8n: renderN8n,
  };

  function demoPathNav(current) {
    var steps = [
      { id: "configuration", label: "Configuration" },
      { id: "delivery", label: "Delivery" },
      { id: "infrastructure", label: "Infrastructure" },
      { id: "cvjobs", label: "CV×Jobs" },
      { id: "spend", label: "Spend" },
    ];
    var links = steps
      .map(function (step) {
        var currentAttr = step.id === current ? ' aria-current="page"' : "";
        return (
          '<button type="button" data-view="' +
          step.id +
          '"' +
          currentAttr +
          ">" +
          step.label +
          "</button>"
        );
      })
      .join("");
    return '<nav class="demo-path" aria-label="Demo path">' + links + "</nav>";
  }

  function setView(id) {
    var view = data.views.find(function (v) {
      return v.id === id;
    });
    if (!view) {
      id = "topology";
      view = data.views[0];
    }
    nav.querySelectorAll("button").forEach(function (btn) {
      if (btn.getAttribute("data-view") === id) {
        btn.setAttribute("aria-current", "page");
      } else {
        btn.removeAttribute("aria-current");
      }
    });
    var onDemoPath =
      id === "topology" ||
      id === "configuration" ||
      id === "delivery" ||
      id === "infrastructure" ||
      id === "cvjobs" ||
      id === "spend";
    panel.innerHTML = (onDemoPath ? demoPathNav(id) : "") + renderers[id]();
    announce(
      view.label +
        (id === "headroom"
          ? " — Cognito Admin + optional n8n demo"
          : id === "litellm"
            ? " — live n8n → LiteLLM probe"
            : id === "cvjobs"
              ? " — Cognito CV×Jobs Matcher↔Editor"
              : " — demo read-only")
    );
    if (location.hash !== "#" + id) {
      try {
        history.replaceState(null, "", "#" + id);
      } catch (err) {
        location.hash = id;
      }
    }
  }

  data.views.forEach(function (view) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = view.label;
    btn.setAttribute("data-view", view.id);
    btn.addEventListener("click", function () {
      setView(view.id);
    });
    nav.appendChild(btn);
  });

  panel.addEventListener("click", function (event) {
    var pathBtn = event.target.closest(".demo-path [data-view]");
    if (pathBtn) {
      setView(pathBtn.getAttribute("data-view"));
      return;
    }
    var priv = event.target.closest(".js-private");
    if (priv) {
      openPrivateModal(priv.getAttribute("data-repo"));
      return;
    }
    var runHr = event.target.closest(".js-headroom-run");
    if (runHr) {
      var boxHr = document.getElementById("headroom-result");
      var urlHr = data.headroom.webhookUrl;
      runHr.disabled = true;
      boxHr.textContent = "Running…";
      announce("Headroom probe started");
      fetch(urlHr, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      })
        .then(function (resp) {
          return resp.text().then(function (text) {
            var parsed = null;
            try {
              parsed = JSON.parse(text);
            } catch (err) {
              parsed = { ok: false, error: text || "non-JSON response" };
            }
            return { okHttp: resp.ok, body: parsed };
          });
        })
        .then(function (out) {
          var b = out.body || {};
          if (!out.okHttp || b.ok === false) {
            boxHr.innerHTML =
              "<p class=\"warn\">Probe failed. No invented savings figure.</p>" +
              "<pre class=\"mono\">" +
              escapeHtml(b.error || JSON.stringify(b, null, 2)) +
              "</pre>";
            announce("Headroom probe failed");
            return;
          }
          boxHr.innerHTML =
            "<div class=\"table-wrap\"><table><caption class=\"visually-hidden\">This click</caption>" +
            "<tbody>" +
            row("tokens_before", b.tokens_before) +
            row("tokens_after", b.tokens_after) +
            row("tokens_saved", b.tokens_saved) +
            row("compression_ratio", b.compression_ratio) +
            row("transforms", JSON.stringify(b.transforms_applied || [])) +
            row("profile", b.savings_profile) +
            row("via", b.via) +
            "</tbody></table></div>";
          announce(
            "Headroom probe done. Saved " +
              String(b.tokens_saved) +
              " tokens"
          );
        })
        .catch(function (err) {
          boxHr.innerHTML =
            "<p class=\"warn\">Request did not complete. No invented savings figure.</p>" +
            "<pre class=\"mono\">" +
            escapeHtml(String(err && err.message ? err.message : err)) +
            "</pre>";
          announce("Headroom probe failed");
        })
        .finally(function () {
          runHr.disabled = false;
        });
      return;
    }

    var runLt = event.target.closest(".js-litellm-run");
    if (!runLt) return;
    var boxLt = document.getElementById("litellm-result");
    var urlLt = data.litellm.webhookUrl;
    runLt.disabled = true;
    boxLt.textContent = "Running…";
    announce("LiteLLM probe started");
    fetch(urlLt, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    })
      .then(function (resp) {
        return resp.text().then(function (text) {
          var parsed = null;
          try {
            parsed = JSON.parse(text);
          } catch (err) {
            parsed = { ok: false, error: text || "non-JSON response" };
          }
          return { okHttp: resp.ok, body: parsed };
        });
      })
      .then(function (out) {
        var b = out.body || {};
        if (!out.okHttp || b.ok === false) {
          boxLt.innerHTML =
            "<p class=\"warn\">Probe failed. No invented health.</p>" +
            "<pre class=\"mono\">" +
            escapeHtml(b.error || JSON.stringify(b, null, 2)) +
            "</pre>";
          announce("LiteLLM probe failed");
          return;
        }
        boxLt.innerHTML =
          "<div class=\"table-wrap\"><table><caption class=\"visually-hidden\">This click</caption>" +
          "<tbody>" +
          row("health", b.health) +
          row("path", b.path) +
          row("via", b.via) +
          "</tbody></table></div>";
        announce("LiteLLM probe done. " + String(b.health || "ok"));
      })
      .catch(function (err) {
        boxLt.innerHTML =
          "<p class=\"warn\">Request did not complete. No invented health.</p>" +
          "<pre class=\"mono\">" +
          escapeHtml(String(err && err.message ? err.message : err)) +
          "</pre>";
        announce("LiteLLM probe failed");
      })
      .finally(function () {
        runLt.disabled = false;
      });
  });

  function row(k, v) {
    return (
      "<tr><th scope=\"row\"><span class=\"mono\">" +
      k +
      "</span></th><td>" +
      escapeHtml(v == null ? "" : String(v)) +
      "</td></tr>"
    );
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  modalClose.addEventListener("click", closePrivateModal);
  modal.addEventListener("click", function (event) {
    if (event.target === modal) closePrivateModal();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !modal.hidden) {
      closePrivateModal();
    }
  });

  var initial = (location.hash || "#topology").slice(1);
  setView(initial);
})();
