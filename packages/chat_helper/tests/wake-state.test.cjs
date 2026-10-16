const wake = require("../../../olympus/wake-state.js");
const { createWidget } = require("../../../olympus/chat.js");
const policy = require("../../../olympus/chat-policy.json");

function fail(message) {
  console.error(message);
  process.exit(1);
}

function fetchImpl(body, ok) {
  return function (url, options) {
    if (url !== "/wake-status") fail("url " + url);
    if (!options || options.cache !== "no-store") fail("missing no-store");
    return Promise.resolve({
      ok: ok !== false,
      json: function () {
        return Promise.resolve(body);
      },
    });
  };
}

function check(body, expectLive, expectState) {
  return wake.readWakeState(fetchImpl(body)).then(function (state) {
    if (state.live !== expectLive) fail("live " + JSON.stringify(state));
    if (state.state !== expectState) fail("state " + JSON.stringify(state));
  });
}

const css = require("fs").readFileSync(require("path").join(__dirname, "../../../olympus/styles.css"), "utf8");
if (!css.includes("color: #9ca3af")) fail("cv grey missing");
if (!css.includes(".chat-label-rag") || !css.includes("color: var(--accent)")) fail("teal rag class missing");
const labelRule = css.slice(css.indexOf(".chat-label {"), css.indexOf(".chat-label-rag"));
if (!labelRule.includes("font-size: 13px")) fail("label is not 13px");
if (labelRule.includes("0.8rem")) fail("label uses rem");
if (!css.includes("color: #8a8a8a")) fail("demo note grey missing");

const consoleJs = require("fs").readFileSync(require("path").join(__dirname, "../../../olympus/console.js"), "utf8") +
  require("fs").readFileSync(require("path").join(__dirname, "../../../olympus/console-data.js"), "utf8");
["admin login", "Sign in", "Headroom", "https://n8n.levkesha.com", "olympusWakeState"].forEach(function (needle) {
  if (consoleJs.indexOf(needle) < 0) fail("console missing " + needle);
});

class Node {
  constructor(doc, tag) {
    this.doc = doc;
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.hidden = false;
    this.disabled = false;
    this._text = "";
    this.parent = null;
  }
  setAttribute(key, value) { this.attrs[key] = value; }
  getAttribute(key) { return this.attrs[key] == null ? null : this.attrs[key]; }
  get textContent() { return this._text + this.children.map(function (child) { return child.textContent || ""; }).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
  appendChild(child) { child.parent = this; this.children.push(child); return child; }
  focus() {}
  addEventListener() {}
  remove() {}
}

function document() {
  const doc = { activeElement: null, listeners: {}, body: null };
  doc.createElement = function (tag) { return new Node(doc, tag); };
  doc.createTextNode = function (text) {
    const node = new Node(doc, "#text");
    node.textContent = text;
    return node;
  };
  doc.addEventListener = function () {};
  doc.body = new Node(doc, "body");
  return doc;
}

global.olympusWakeState = wake;
let seen = 0;
const widget = createWidget(document(), {
  policy: policy,
  wakeFetch: function (url, options) {
    seen += 1;
    return fetchImpl({ state: "live", approved: true })(url, options);
  },
});

Promise.resolve()
  .then(function () { return check({ state: "live" }, true, "live"); })
  .then(function () { return check({ state: "waking" }, false, "waking"); })
  .then(function () { return check({ state: "nope" }, false, "asleep"); })
  .then(function () { return check(null, false, "asleep"); })
  .then(function () { return wake.readWakeState(fetchImpl({}, false)); })
  .then(function (state) {
    if (state.live !== false || state.state !== "asleep") fail("http failure was not asleep");
  })
  .then(function () { return widget.wakeReady; })
  .then(function (state) {
    if (seen !== 1) fail("chat did not read the shared wake state");
    if (!state || state.live !== true) fail("chat wake state was not live");
  })
  .catch(function (error) {
    fail(error && error.message ? error.message : String(error));
  });
