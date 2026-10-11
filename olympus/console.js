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
      '<p><a class="btn" href="' +
      escapeHtml(cj.ctaHref) +
      '">' +
      escapeHtml(cj.cta) +
      "</a></p></section>"
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
      '<ul class="panel-lines">' +
      lines +
      "</ul></section>"
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
      '<p class="sleep-pill"><span class="sleep-dot"></span>' +
      escapeHtml(n.pill) +
      "</p>" +
      '<div class="flow-cards">' +
      cards +
      "</div>" +
      '<p><a class="btn btn-primary" href="' +
      escapeHtml(n.ctaHref) +
      '">' +
      escapeHtml(n.cta) +
      "</a></p></section>"
    );
  }

  var renderers = {
    glossary: renderGlossary,
    cvjobs: renderCvJobs,
    spend: renderSpend,
    n8n: renderN8n,
  };

  function setView(id) {
    var view = data.views.find(function (v) {
      return v.id === id;
    });
    if (!view || !renderers[id]) {
      view = data.views[0];
      id = view.id;
    }
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
})();
