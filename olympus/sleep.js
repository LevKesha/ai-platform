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

  function wakeWithin() {
    var data = window.OLYMPUS_CONSOLE;
    if (!data || !data.wakeWithin) throw new Error("OLYMPUS_CONSOLE.wakeWithin missing");
    return data.wakeWithin;
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

  function drawPoster() {
    poster.textContent = "";
  }

  function statusLabel(kind) {
    if (kind === "is-live") return "Live";
    if (kind === "is-wait") return "Waiting";
    return "Asleep";
  }

  function pill(kind, text) {
    var node = el("p", "sleep-pill");
    node.appendChild(el("span", "sleep-dot" + (kind ? " " + kind : "")));
    node.appendChild(el("span", "sleep-status", statusLabel(kind)));
    node.appendChild(document.createTextNode(text));
    return node;
  }

  function show(name, opts) {
    opts = opts || {};
    panel.textContent = "";
    drawPoster();
    if (name === "asleep") renderAsleep();
    else if (name === "requested") renderRequested();
    else if (name === "scheduled") renderScheduled(opts.wakeAt);
    else if (name === "waking") renderWaking(opts.approved === true);
    else if (name === "live") renderLive();
    else renderForm(opts.where || "");
    if (live) live.textContent = "Console state " + name;
  }

  function renderAsleep() {
    var figure = el("figure", "panel-diagram");
    var img = document.createElement("img");
    img.src = "diagrams/soft-down.png";
    var caption = "You pick a time. Lev approves. It starts then and is live in " + wakeWithin() + ".";
    img.className = "sleep-strip-desk";
    img.alt = caption + " Sleeps after 1h idle.";
    img.width = 1014;
    img.height = 203;
    img.loading = "lazy";
    var phone = document.createElement("img");
    phone.className = "sleep-strip-phone";
    phone.src = "diagrams/soft-down-phone.png";
    phone.alt = caption + " Sleeps after 1h idle.";
    phone.width = 360;
    phone.height = 560;
    phone.loading = "lazy";
    figure.appendChild(img);
    figure.appendChild(phone);
    panel.appendChild(figure);
    panel.appendChild(pill("", "Asleep. Request a look."));
    var button = el("button", "btn btn-primary sleep-cta is-marked", "Request a look");
    button.type = "button";
    button.addEventListener("click", function () {
      show("form", { where: params.get("where") || cookieWhere() });
    });
    panel.appendChild(button);
  }

  function renderRequested() {
    panel.appendChild(pill("is-wait", "Sent. Lev replies by email."));
    panel.appendChild(el("h2", null, "The console, live"));
  }

  function renderWaking(approved) {
    panel.appendChild(pill("is-wait", "Waking up. This takes " + wakeWithin() + ". We'll email you when it's live."));
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
      ["ok", "Servers"],
      ["run", "Sign-in"],
      ["wait", "AI gateway"],
      ["wait", "Console"]
    ].forEach(function (step) {
      var item = el("li");
      if (step[0] === "ok") {
        item.appendChild(el("span", "sleep-ok", "\u2713"));
        item.appendChild(el("span", "sleep-status", "Done"));
      } else if (step[0] === "run") {
        var spin = el("span", "sleep-spin");
        spin.setAttribute("aria-hidden", "true");
        item.appendChild(spin);
        item.appendChild(el("span", "sleep-spin-text", "Waking\u2026"));
      } else {
        item.appendChild(el("span", "sleep-pend", "\u25cb"));
        item.appendChild(el("span", "sleep-status", "Waiting"));
      }
      item.appendChild(el("span", step[0] === "wait" ? "sleep-muted" : null, step[1]));
      list.appendChild(item);
    });
    panel.appendChild(list);
  }

  function renderScheduled(wakeAt) {
    var when = wakeAt ? israelClock(Number(wakeAt)) : "";
    panel.appendChild(pill("is-wait", when ? "Scheduled for " + when + " Israel time." : "Scheduled."));
    panel.appendChild(el("h2", null, "The console, live"));
  }

  function renderLive() {
    panel.appendChild(pill("is-live", "Live now. Sleeps after an hour idle."));
    panel.appendChild(el("h2", null, "The console, live"));
    var open = el("a", "btn btn-primary sleep-cta", "Open live demo");
    open.href = demoHref;
    panel.appendChild(open);
    panel.appendChild(el("p", "sleep-tiny", "Interview booked? Lev wakes it ahead of time."));
  }

  function two(n) {
    n = Math.floor(Math.abs(n));
    return (n < 10 ? "0" : "") + n;
  }

  function jerusalemParts(ms) {
    var fmt = new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Jerusalem",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23"
    });
    var bag = {};
    fmt.formatToParts(new Date(ms)).forEach(function (part) {
      if (part.type !== "literal") bag[part.type] = part.value;
    });
    return bag;
  }

  function israelClock(ms) {
    var parts = jerusalemParts(ms);
    return parts.day + "/" + parts.month + "/" + parts.year + " " + parts.hour + ":" + parts.minute;
  }

  function israelIso(localValue) {
    var match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(localValue || "");
    if (!match) return "";
    var want = match[1] + "-" + match[2] + "-" + match[3] + " " + match[4] + ":" + match[5];
    var guess = Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], 0);
    var i;
    for (i = 0; i < 4; i++) {
      var parts = jerusalemParts(guess);
      var got = parts.year + "-" + parts.month + "-" + parts.day + " " + parts.hour + ":" + parts.minute;
      if (got === want) break;
      var gotMs = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute, 0);
      var wantMs = Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], 0);
      guess += wantMs - gotMs;
    }
    var shown = jerusalemParts(guess);
    var shownKey = shown.year + "-" + shown.month + "-" + shown.day + " " + shown.hour + ":" + shown.minute;
    if (shownKey !== want) return "";
    var localAsUtc = Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], 0);
    var offMin = Math.round((localAsUtc - guess) / 60000);
    var sign = offMin >= 0 ? "+" : "-";
    return match[1] + "-" + match[2] + "-" + match[3] + "T" + match[4] + ":" + match[5] + ":00" + sign + two(offMin / 60) + ":" + two(offMin % 60);
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
    form.appendChild(el("h2", "sleep-form-title", "See it live"));
    var rowName = el("div", "sleep-row");
    rowName.appendChild(field("Name", "text", "name", true));
    rowName.appendChild(field("Company", "text", "company", true));
    var rowMail = el("div", "sleep-row");
    rowMail.appendChild(field("Email", "email", "email", true));
    form.appendChild(rowName);
    form.appendChild(rowMail);
    var whenLabel = el("label");
    whenLabel.appendChild(document.createTextNode("Pick a time"));
    var whenInput = document.createElement("input");
    whenInput.type = "datetime-local";
    whenInput.name = "when";
    whenInput.required = where === "IL";
    whenLabel.appendChild(whenInput);
    var whenWrap = el("div");
    whenWrap.appendChild(whenLabel);
    whenWrap.appendChild(el("p", "sleep-tiny", "Israel time. Now " + israelClock(Date.now()) + "."));
    whenWrap.hidden = where === "elsewhere";
    form.appendChild(whenWrap);
    var legend = el("span", "sleep-muted", "Where are you?");
    form.appendChild(legend);
    var radios = el("div", "sleep-radios");
    radios.appendChild(radio("IL", "Israel", where === "IL"));
    radios.appendChild(radio("elsewhere", "Elsewhere", where === "elsewhere"));
    radios.addEventListener("change", function () {
      var picked = form.querySelector("input[name=where]:checked");
      var il = !!(picked && picked.value === "IL");
      whenWrap.hidden = !il;
      whenInput.required = il;
    });
    form.appendChild(radios);
    var error = el("p", "sleep-error");
    error.hidden = true;
    form.appendChild(error);
    var send = el("button", "btn btn-primary sleep-cta", "Request");
    send.type = "submit";
    form.appendChild(send);
    form.appendChild(el("p", "sleep-tiny", "Lev reads each one."));
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
      where: "",
      form_token: "wake"
    };
    var chosen = form.querySelector("input[name=where]:checked");
    data.where = chosen ? chosen.value : "";
    data.kind = data.where === "IL" ? "wake" : "walkthrough";
    if (!data.name || !data.company || !data.email || !data.where) {
      error.hidden = false;
      error.textContent = "Name, company, email, and where you are are required.";
      return;
    }
    if (data.where === "IL") {
      data.when = israelIso(form.when.value);
      var at = Date.parse(data.when);
      var minAhead = 5 * 60 * 1000;
      var maxAhead = 7 * 24 * 60 * 60 * 1000;
      if (!data.when || !(at >= Date.now() + minAhead) || at > Date.now() + maxAhead) {
        error.hidden = false;
        error.textContent = "Pick an Israel time at least 5 minutes ahead, and within 7 days.";
        return;
      }
    } else {
      data.when = "";
    }
    error.hidden = true;
    if (!endpoint) {
      error.hidden = false;
      error.textContent = "This draft has no wake address yet, so nothing was sent.";
      return;
    }
    var send = form.querySelector("button[type=submit]");
    if (send) send.disabled = true;
    function showSendError() {
      if (send) send.disabled = false;
      error.hidden = false;
      error.textContent = "The request did not send. Lev has not received it.";
    }
    fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(data)
    }).then(function (res) {
      var type = res && res.headers && typeof res.headers.get === "function" ? String(res.headers.get("content-type") || "") : "";
      if (!res || !res.ok || res.status === 403 || res.status === 404 || type.indexOf("text/html") !== -1) {
        showSendError();
        return;
      }
      show("requested");
    }).catch(showSendError);
  }

  function allowed(name) {
    if (name === "asleep" || name === "form") return true;
    return preview && (name === "requested" || name === "scheduled" || name === "waking" || name === "live");
  }

  var asked = params.get("state") || "";
  var where = params.get("where") || cookieWhere();
  if (where !== "IL" && where !== "elsewhere") where = "";
  if (allowed(asked)) {
    show(asked, { where: where, approved: preview && params.get("approved") === "1" });
  } else {
    show("asleep");
  }

  if (!preview && (!asked || asked === "asleep") && window.olympusWakeState) {
    window.olympusWakeState.readWakeState().then(function (body) {
      if (!body || (body.state !== "requested" && body.state !== "scheduled" && body.state !== "waking" && body.state !== "live")) return;
      show(body.state, { approved: body.approved === true, wakeAt: body.wakeAt });
    });
  }
})();
