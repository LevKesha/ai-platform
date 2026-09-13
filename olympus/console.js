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

  function badge(icon, label) {
    return (
      '<span class="badge"><span class="badge-icon" aria-hidden="true">' +
      icon +
      "</span> " +
      label +
      "</span>"
    );
  }

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

  function renderTopology() {
    var rows = data.topology.services
      .map(function (svc) {
        var edge = svc.public
          ? badge("↗", "Public ALB") +
            ' <a href="' +
            svc.url +
            '" rel="noopener noreferrer" target="_blank">' +
            svc.url.replace("https://", "") +
            ' <span class="visually-hidden">(opens in new tab)</span></a>'
          : badge("⬡", "ClusterIP") +
            " <span class=\"muted\">" +
            data.empty.noPublicEndpoint +
            "</span>";
        return (
          "<tr><th scope=\"row\"><span class=\"mono\">" +
          svc.name +
          "</span></th><td>" +
          svc.role +
          "</td><td>" +
          edge +
          "</td></tr>"
        );
      })
      .join("");

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
      "<div class=\"empty\"><strong>n8n unavailable</strong>" +
      data.empty.n8nUnavailable +
      "</div>" +
      "<div class=\"table-wrap\"><table><caption class=\"visually-hidden\">Platform services</caption>" +
      "<thead><tr><th scope=\"col\">Service</th><th scope=\"col\">Role</th><th scope=\"col\">Edge</th></tr></thead>" +
      "<tbody>" +
      rows +
      "</tbody></table></div>" +
      "<h2>Private evidence</h2>" +
      "<p class=\"muted\">No GitHub links. Open a name for the private-repo note.</p>" +
      "<p>" +
      priv +
      "</p>"
    );
  }

  function renderConfiguration() {
    var cfg = data.configuration;
    var notes = cfg.notes.map(function (n) {
      return "<li>" + n + "</li>";
    }).join("");
    return (
      "<h1>Configuration</h1>" +
      "<p class=\"lede\">Bedrock SSOT is a file in the public ai-platform repo. This panel is read-only.</p>" +
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
      "</dl><ul>" +
      notes +
      "</ul></article>"
    );
  }

  function renderDelivery() {
    var d = data.delivery;
    return (
      "<h1>Delivery</h1>" +
      "<p class=\"lede\">Reusable GHA pattern only. No mutate controls.</p>" +
      "<article class=\"card\"><p>" +
      d.cicd +
      "</p><p>" +
      d.infraWorkflows +
      "</p><p class=\"muted\">" +
      d.note +
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
      "<dt>Production cluster</dt><dd>none</dd>" +
      "</dl></article>" +
      "<p><button type=\"button\" class=\"btn js-private\" data-repo=\"infrastructure\">Why no GitHub link?</button></p>"
    );
  }

  function renderSpend() {
    var s = data.spend;
    var rows = s.surfaces
      .map(function (row) {
        return (
          "<tr><th scope=\"row\">" +
          row.name +
          "</th><td>" +
          row.access +
          "</td><td>" +
          row.note +
          "</td></tr>"
        );
      })
      .join("");
    var hops = s.hops
      .map(function (h) {
        return (
          "<tr><th scope=\"row\">" +
          h.name +
          "</th><td>" +
          h.path +
          "</td></tr>"
        );
      })
      .join("");
    return (
      "<h1>Services &amp; Spend</h1>" +
      "<div class=\"empty\"><strong>Demo data</strong>" +
      data.empty.demoData +
      "</div>" +
      "<div class=\"empty\"><strong>No public endpoint</strong>" +
      data.empty.noPublicEndpoint +
      "</div>" +
      "<h2>Internal surfaces</h2>" +
      "<div class=\"table-wrap\"><table><caption class=\"visually-hidden\">Internal AI cost surfaces</caption>" +
      "<thead><tr><th scope=\"col\">Surface</th><th scope=\"col\">Access</th><th scope=\"col\">Honesty</th></tr></thead>" +
      "<tbody>" +
      rows +
      "</tbody></table></div>" +
      "<h2>Proxy hops</h2>" +
      "<div class=\"table-wrap\"><table><caption class=\"visually-hidden\">CLI vs platform proxy</caption>" +
      "<thead><tr><th scope=\"col\">Hop</th><th scope=\"col\">Path</th></tr></thead>" +
      "<tbody>" +
      hops +
      "</tbody></table></div>"
    );
  }

  var renderers = {
    topology: renderTopology,
    configuration: renderConfiguration,
    delivery: renderDelivery,
    infrastructure: renderInfrastructure,
    spend: renderSpend,
  };

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
    panel.innerHTML = renderers[id]();
    announce(view.label + " — demo read-only");
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
    var btn = event.target.closest(".js-private");
    if (!btn) return;
    openPrivateModal(btn.getAttribute("data-repo"));
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

  var initial = (location.hash || "#topology").slice(1);
  setView(initial);
})();
