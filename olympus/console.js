(function () {
  var data = window.OLYMPUS_CONSOLE;
  var nav = document.getElementById("console-nav");
  var panel = document.getElementById("console-panel");
  var live = document.getElementById("console-live");
  var modal = document.getElementById("private-modal");
  var modalBody = document.getElementById("private-modal-body");
  var modalClose = document.getElementById("private-modal-close");
  var lastFocus = null;
  var glossaryMarkup = panel ? panel.innerHTML : "";

  if (!data || !nav || !panel) return;

  function announce(text) {
    if (live) live.textContent = text;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
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

  function renderGlossary() {
    return glossaryMarkup;
  }

  var clusterLive = false;
  var viewId = "glossary";

  function requestLook(href) {
    return (
      '<p><a class="btn btn-primary" href="' +
      escapeHtml(href) +
      '">Request a look</a></p>'
    );
  }

  function renderCvJobs() {
    var cj = data.cvjobs;
    var badges = (cj.badges || [])
      .map(function (badge) {
        return '<span class="badge">' + escapeHtml(badge) + "</span>";
      })
      .join(" ");
    return (
      '<section class="hub-glossary" aria-labelledby="pv-t">' +
      '<h2 id="pv-t">' +
      escapeHtml(cj.title) +
      "</h2>" +
      '<p class="console-subline">' +
      escapeHtml(cj.subline) +
      "</p>" +
      '<p class="panel-tag">' +
      badges +
      "</p>" +
      '<figure class="panel-diagram"><img src="diagrams/cv-jobs-flow.svg" alt="CV×Jobs: a master CV, the jobs you paste, a fit report, a tailored CV, and an export, with a model call and a role focus." width="1016" height="444" loading="lazy"></figure>' +
      '<p><a class="btn" href="' +
      escapeHtml(cj.ctaHref) +
      '">' +
      escapeHtml(cj.cta) +
      "</a></p>" +
      requestLook(cj.lookHref) +
      "</section>"
    );
  }

  function renderSpend() {
    var s = data.spend;
    var lines = (s.lines || [])
      .map(function (row) {
        return (
          "<li><strong>" +
          escapeHtml(row.label) +
          "</strong>: " +
          escapeHtml(row.text) +
          "</li>"
        );
      })
      .join("");
    return (
      '<section class="hub-glossary" aria-labelledby="pv-t">' +
      '<h2 id="pv-t">' +
      escapeHtml(s.title) +
      "</h2>" +
      '<p class="console-subline">' +
      escapeHtml(s.subline) +
      "</p>" +
      '<p class="panel-tag"><span class="badge">' +
      escapeHtml(s.badge) +
      "</span></p>" +
      '<figure class="panel-diagram"><img src="diagrams/spend-loop.svg" alt="Spend: CV×Jobs sends requests through the LiteLLM gateway. n8n calls Headroom to compress. Spend is logged in one place." width="1016" height="444" loading="lazy"></figure>' +
      '<ul class="panel-lines">' +
      lines +
      "</ul>" +
      requestLook(s.lookHref) +
      "</section>"
    );
  }

  function renderN8n() {
    var n = data.n8n;
    var cards = (n.workflows || [])
      .map(function (flow) {
        return (
          '<article class="card"><h3>' +
          escapeHtml(flow.label) +
          '</h3><p class="mono">' +
          escapeHtml(flow.name) +
          "</p></article>"
        );
      })
      .join("");
    return (
      '<section class="hub-glossary" aria-labelledby="pv-t">' +
      '<h2 id="pv-t">' +
      escapeHtml(n.title) +
      "</h2>" +
      '<p class="console-subline">' +
      escapeHtml(n.subline) +
      "</p>" +
      '<p class="sleep-pill"><span class="sleep-dot"></span><span class="sleep-status">Asleep</span> ' +
      escapeHtml(n.pill) +
      "</p>" +
      '<div class="flow-cards">' +
      cards +
      "</div>" +
      requestLook(n.ctaHref) +
      "</section>"
    );
  }

  function renderLiteLLM() {
    var lt = data.litellm;
    return (
      '<section class="hub-glossary" aria-labelledby="pv-t">' +
      '<h2 id="pv-t">' +
      escapeHtml(lt.title) +
      "</h2>" +
      '<p class="console-subline">' +
      escapeHtml(lt.subline) +
      "</p>" +
      requestLook(lt.lookHref) +
      "</section>"
    );
  }

  function renderDashboards() {
    var board = data.dashboards;
    var body;
    if (!clusterLive) {
      body =
        '<p><a href="' +
        escapeHtml(board.ctaHref) +
        '">' +
        escapeHtml(board.cta) +
        "</a></p>";
    } else {
      body =
        '<ul class="dash-tiles">' +
        (board.tiles || [])
          .map(function (tile) {
            return (
              '<li><div class="dash-tile"><span class="dash-name">' +
              escapeHtml(tile.label) +
              '</span><a class="btn btn-primary" href="' +
              escapeHtml(tile.href) +
              '">' +
              escapeHtml(tile.login) +
              "</a></div></li>"
            );
          })
          .join("") +
        "</ul>";
    }
    return (
      '<section class="hub-glossary" aria-labelledby="pv-t">' +
      '<h2 id="pv-t">' +
      escapeHtml(board.title) +
      "</h2>" +
      '<p class="demo-note">' +
      escapeHtml(board.note) +
      "</p>" +
      body +
      "</section>"
    );
  }

  var renderers = {
    glossary: renderGlossary,
    cvjobs: renderCvJobs,
    spend: renderSpend,
    n8n: renderN8n,
    litellm: renderLiteLLM,
    dashboards: renderDashboards,
  };

  function setView(id) {
    var view = data.views.find(function (v) {
      return v.id === id;
    });
    if (!view || !renderers[id]) {
      view = data.views[0];
      id = view.id;
    }
    viewId = id;
    nav.querySelectorAll("button").forEach(function (btn) {
      if (btn.getAttribute("data-view") === id) {
        btn.setAttribute("aria-current", "page");
      } else {
        btn.removeAttribute("aria-current");
      }
    });
    panel.innerHTML = renderers[id]();
    announce(view.label);
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
    var priv = event.target.closest(".js-private");
    if (!priv) return;
    openPrivateModal(priv.getAttribute("data-repo"));
  });

  modalClose.addEventListener("click", closePrivateModal);
  modal.addEventListener("click", function (event) {
    if (event.target === modal) closePrivateModal();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !modal.hidden) {
      closePrivateModal();
    }
  });

  var initial = (location.hash || "").replace(/^#/, "");
  if (initial) {
    setView(initial);
  } else {
    var first = nav.querySelector("button");
    if (first) first.setAttribute("aria-current", "page");
  }

  function applyWake(state) {
    var liveNow = !!(state && state.live);
    if (liveNow === clusterLive) return;
    clusterLive = liveNow;
    if (viewId === "dashboards") setView(viewId);
  }

  if (window.olympusWakeState) {
    window.olympusWakeState.readWakeState().then(applyWake);
  }
})();
