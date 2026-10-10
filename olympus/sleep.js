(function () {
  var params = new URLSearchParams(window.location.search);

  function cookieWhere() {
    var parts = document.cookie ? document.cookie.split(";") : [];
    var i;
    for (i = 0; i < parts.length; i++) {
      var bit = parts[i].replace(/^\s+/, "");
      var prefix = "olympus-where=";
      if (bit.indexOf(prefix) === 0) return decodeURIComponent(bit.slice(prefix.length));
    }
    return "";
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  var gated = document.getElementById("gated-panel");
  if (gated) {
    var part = params.get("part") || "";
    if (/^[a-z0-9-]{1,24}$/.test(part)) {
      var title = document.getElementById("gated-title");
      if (title) title.textContent = "Console · " + part;
    }
    var look = document.getElementById("gated-look");
    if (look) {
      look.addEventListener("click", function () {
        window.location.href = "console.html?state=form";
      });
    }
    return;
  }

  var root = document.getElementById("sleep-portal");
  if (!root) return;
  var panel = document.getElementById("sleep-panel");
  var poster = document.getElementById("sleep-poster");
  var live = document.getElementById("sleep-live");
  var endpoint = root.getAttribute("data-endpoint") || "";
  var demoHref = root.getAttribute("data-demo") || "/cv-jobs/";
  var preview = params.get("preview") === "1";
  if (preview) document.body.classList.add("sleep-shot");

  function bars(liveFrame) {
    var frame = el("div", liveFrame ? "sleep-ui is-live" : "sleep-ui");
    ["40%", "100%", "80%", "70px", "60%"].forEach(function (hint, index) {
      var bar = document.createElement("div");
      if (index === 3) bar.style.height = hint;
      else if (hint !== "100%") bar.style.width = hint;
      frame.appendChild(bar);
    });
    return frame;
  }

  function drawPoster(isLive) {
    poster.textContent = "";
    if (isLive) {
      var wrap = el("div", "sleep-live-frame");
      wrap.appendChild(bars(true));
      poster.appendChild(wrap);
      return;
    }
    poster.appendChild(bars(false));
    poster.appendChild(el("span", "sleep-cap", "Captured [date] on a live run."));
  }

  function pill(kind, text) {
    var node = el("p", "sleep-pill");
    node.appendChild(el("span", "sleep-dot" + (kind ? " " + kind : "")));
    node.appendChild(document.createTextNode(text));
    return node;
  }

  function show(name, opts) {
    opts = opts || {};
    panel.textContent = "";
    drawPoster(name === "live");
    if (name === "asleep") renderAsleep();
    else if (name === "requested") renderRequested();
    else if (name === "waking") renderWaking(opts.approved === true);
    else if (name === "live") renderLive();
    else renderForm(opts.where || "");
    if (live) live.textContent = "Console state " + name;
  }

  function renderAsleep() {
    panel.appendChild(pill("", "Asleep. Request a look."));
    panel.appendChild(el("h2", null, "The console, live"));
    var button = el("button", "btn btn-primary sleep-cta is-marked", "Request a look");
    button.type = "button";
    button.addEventListener("click", function () {
      show("form", { where: params.get("where") || cookieWhere() });
    });
    panel.appendChild(button);
  }

  function renderRequested() {
    panel.appendChild(pill("is-wait", "Request sent. Lev will reply by email."));
    panel.appendChild(el("h2", null, "The console, live"));
  }

  function renderWaking(approved) {
    panel.appendChild(pill("is-wait", "Waking up. This takes about 15 minutes. We'll email you when it's live."));
    panel.appendChild(el("h2", null, "The console, live"));
    if (approved) {
      var bar = el("div", "sleep-bar");
      bar.setAttribute("role", "progressbar");
      bar.setAttribute("aria-valuemin", "0");
      bar.setAttribute("aria-valuemax", "100");
      bar.setAttribute("aria-valuenow", "40");
      bar.setAttribute("aria-label", "Wake progress");
      bar.appendChild(document.createElement("i"));
      panel.appendChild(bar);
    }
    var list = el("ul", "sleep-steps");
    [
      ["ok", "Cluster nodes"],
      ["run", "Gateway & auth"],
      ["wait", "LiteLLM"],
      ["wait", "Console"]
    ].forEach(function (step) {
      var item = el("li");
      if (step[0] === "ok") item.appendChild(el("span", "sleep-ok", "\u2713"));
      else if (step[0] === "run") {
        var spin = el("span", "sleep-spin");
        spin.setAttribute("aria-hidden", "true");
        item.appendChild(spin);
      } else item.appendChild(el("span", "sleep-pend", "\u25cb"));
      item.appendChild(el("span", step[0] === "wait" ? "sleep-muted" : null, step[1]));
      list.appendChild(item);
    });
    panel.appendChild(list);
  }

  function renderLive() {
    panel.appendChild(pill("is-live", "Live now. Sleeps after an hour idle."));
    panel.appendChild(el("h2", null, "The console, live"));
    var open = el("a", "btn btn-primary sleep-cta", "Open live demo");
    open.href = demoHref;
    panel.appendChild(open);
    panel.appendChild(el("p", "sleep-tiny", "Interview booked? Lev wakes it ahead of time."));
  }

  function field(labelText, type, name, required) {
    var label = el("label");
    label.appendChild(document.createTextNode(labelText));
    var input = document.createElement("input");
    input.type = type;
    input.name = name;
    input.autocomplete = name === "email" ? "email" : "name";
    if (required) input.required = true;
    label.appendChild(input);
    return label;
  }

  function renderForm(where) {
    var form = el("form", "sleep-form");
    form.noValidate = true;
    form.appendChild(el("h2", "sleep-form-title", "See it running"));
    var rowName = el("div", "sleep-row");
    rowName.appendChild(field("Name", "text", "name", true));
    rowName.appendChild(field("Company", "text", "company", true));
    var rowMail = el("div", "sleep-row");
    rowMail.appendChild(field("Email", "email", "email", true));
    var when = field("When works for you (optional)", "text", "when", false);
    rowMail.appendChild(when);
    form.appendChild(rowName);
    form.appendChild(rowMail);
    var legend = el("span", "sleep-muted", "Where are you?");
    form.appendChild(legend);
    var radios = el("div", "sleep-radios");
    radios.appendChild(radio("IL", "Israel", where === "IL"));
    radios.appendChild(radio("elsewhere", "Elsewhere", where === "elsewhere"));
    form.appendChild(radios);
    var error = el("p", "sleep-error");
    error.hidden = true;
    form.appendChild(error);
    var send = el("button", "btn btn-primary sleep-cta", "Send request");
    send.type = "submit";
    form.appendChild(send);
    form.appendChild(el("p", "sleep-tiny", "Lev reads every request himself."));
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submitForm(form, error);
    });
    panel.appendChild(form);
    var nameInput = form.querySelector("input[name=name]");
    if (nameInput) nameInput.focus();
  }

  function radio(value, labelText, checked) {
    var label = el("label");
    var input = document.createElement("input");
    input.type = "radio";
    input.name = "where";
    input.value = value;
    input.checked = checked;
    label.appendChild(input);
    label.appendChild(document.createTextNode(labelText));
    return label;
  }

  function submitForm(form, error) {
    var data = {
      name: form.name.value.replace(/^\s+|\s+$/g, ""),
      company: form.company.value.replace(/^\s+|\s+$/g, ""),
      email: form.email.value.replace(/^\s+|\s+$/g, ""),
      when: form.when.value.replace(/^\s+|\s+$/g, ""),
      where: ""
    };
    var chosen = form.querySelector("input[name=where]:checked");
    data.where = chosen ? chosen.value : "";
    data.kind = data.where === "IL" ? "wake" : "walkthrough";
    if (!data.name || !data.company || !data.email || !data.where) {
      error.hidden = false;
      error.textContent = "Name, company, email, and where you are are required.";
      return;
    }
    error.hidden = true;
    if (!endpoint) {
      error.hidden = false;
      error.textContent = "This draft has no wake address yet, so nothing was sent.";
      return;
    }
    var send = form.querySelector("button[type=submit]");
    if (send) send.disabled = true;
    fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(data)
    }).then(function (res) {
      if (!res.ok) throw new Error("status " + res.status);
      show("requested");
    }).catch(function () {
      if (send) send.disabled = false;
      error.hidden = false;
      error.textContent = "The request did not send. Lev has not received it.";
    });
  }

  function allowed(name) {
    if (name === "asleep" || name === "form") return true;
    return preview && (name === "requested" || name === "waking" || name === "live");
  }

  var asked = params.get("state") || "";
  var where = params.get("where") || cookieWhere();
  if (where !== "IL" && where !== "elsewhere") where = "";
  if (allowed(asked)) {
    show(asked, { where: where, approved: preview && params.get("approved") === "1" });
  } else {
    show("asleep");
  }

  if (!preview && (!asked || asked === "asleep")) {
    fetch("/wake-status", { cache: "no-store" }).then(function (res) {
      if (!res.ok) return null;
      return res.json();
    }).then(function (body) {
      if (!body || (body.state !== "requested" && body.state !== "waking" && body.state !== "live")) return;
      show(body.state, { approved: body.approved === true });
    }).catch(function () {});
  }
})();
