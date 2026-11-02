"use strict";

// A 421 response is the only status that shows the blocked sentence. The body is
// never read. 403 and 404 keep their existing handling in both the chat and the
// wake form.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.join(__dirname, "../../..");
const wake = require(path.join(root, "olympus/wake-state.js"));
const { createWidget } = require(path.join(root, "olympus/chat.js"));
const policy = require(path.join(root, "olympus/chat-policy.json"));

const SENTENCE = "This part is open to visitors in Israel only.";
const HTML_BODY = "<html><body><script>alert(1)</script>Blocked</body></html>";

function fail(message) {
  console.error(message);
  process.exit(1);
}

function flush() {
  return new Promise(function (resolve) {
    setTimeout(resolve, 20);
  });
}

class Node {
  constructor(doc, tag) {
    this.doc = doc;
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.listeners = {};
    this.parent = null;
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.checked = false;
    this._text = "";
  }
  setAttribute(key, value) {
    this.attrs[key] = String(value);
  }
  getAttribute(key) {
    return this.attrs[key] == null ? null : this.attrs[key];
  }
  get className() {
    return this.attrs.class || "";
  }
  set className(value) {
    this.attrs.class = String(value);
  }
  get textContent() {
    return this._text + this.children.map(function (child) { return child.textContent || ""; }).join("");
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  set innerHTML(value) {
    fail("innerHTML was written: " + value);
  }
  appendChild(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  insertBefore(child, ref) {
    child.parent = this;
    const at = this.children.indexOf(ref);
    if (at < 0) this.children.push(child);
    else this.children.splice(at, 0, child);
    return child;
  }
  remove() {
    if (!this.parent) return;
    this.parent.children = this.parent.children.filter(function (child) { return child !== this; }, this);
    this.parent = null;
  }
  addEventListener(type, fn) {
    (this.listeners[type] || (this.listeners[type] = [])).push(fn);
  }
  focus() {
    this.doc.activeElement = this;
  }
  all() {
    const out = [];
    (function walk(node) {
      node.children.forEach(function (child) {
        out.push(child);
        walk(child);
      });
    })(this);
    return out;
  }
  querySelector(selector) {
    const match = /^(\w+)(?:\[name=(\w+)\])?(?:\[type=(\w+)\])?(:checked)?$/.exec(selector);
    if (!match) fail("unsupported selector " + selector);
    return this.all().filter(function (node) {
      if (node.tag !== match[1]) return false;
      if (match[2] && node.name !== match[2]) return false;
      if (match[3] && node.type !== match[3]) return false;
      if (match[4] && !node.checked) return false;
      return true;
    })[0] || null;
  }
}

function makeDocument() {
  const doc = { activeElement: null, listeners: {}, cookie: "" };
  doc.createElement = function (tag) { return new Node(doc, tag); };
  doc.createTextNode = function (text) {
    const node = new Node(doc, "#text");
    node.textContent = text;
    return node;
  };
  doc.addEventListener = function (type, fn) {
    (doc.listeners[type] || (doc.listeners[type] = [])).push(fn);
  };
  doc.body = new Node(doc, "body");
  return doc;
}

function response(status) {
  return {
    status: status,
    ok: status >= 200 && status < 300,
    headers: { get: function () { return "text/html; charset=utf-8"; } },
    json: function () {
      if (status === 421) fail("421 body was parsed");
      return Promise.reject(new Error("not json"));
    },
    text: function () {
      if (status === 421) fail("421 body was read");
      return Promise.resolve(HTML_BODY);
    },
  };
}

function hasBodyLeak(text) {
  return text.indexOf("alert(1)") !== -1 || text.indexOf("<script") !== -1;
}

// Shared helper.
if (wake.blockedMessage(response(421)) !== SENTENCE) fail("helper did not return the sentence for 421");
[200, 403, 404, 500, undefined, null].forEach(function (status) {
  const res = status == null ? status : response(status);
  if (wake.blockedMessage(res) !== "") fail("helper fired for " + status);
});

// Chat panel.
function chatSubmit(status, typedText) {
  const doc = makeDocument();
  const widget = createWidget(doc, {
    policy: policy,
    fetchImpl: function () { return Promise.resolve(response(status)); },
  });
  widget.open();
  widget.input.value = typedText;
  const form = widget.input.parent;
  (form.listeners.submit || []).forEach(function (fn) { fn({ preventDefault: function () {} }); });
  return flush().then(function () { return { doc: doc, widget: widget }; });
}

function assertChatBlocked(page) {
  const widget = page.widget;
  const notes = widget.log.all().filter(function (node) { return node.getAttribute("role") === "status"; });
  if (notes.length !== 1) fail("chat 421 status slots: " + notes.length);
  const note = notes[0];
  if (note.textContent !== SENTENCE) fail("chat 421 sentence was " + JSON.stringify(note.textContent));
  if (note.children.length !== 0) fail("chat 421 status slot has child nodes");
  if (note.className !== "chat-note") fail("chat 421 note class " + note.className);
  if (widget.log.all().some(function (node) { return node.tag === "a" || node.className === "chat-closed"; })) {
    fail("chat 421 rendered a link or the closed card");
  }
  if (widget.log.textContent.indexOf(policy.closed) !== -1) fail("chat 421 showed the closed copy");
  if (widget.log.textContent.indexOf(policy.retry) !== -1) fail("chat 421 showed the retry copy");
  if (hasBodyLeak(widget.log.textContent)) fail("chat 421 body leaked into the panel");
  if (widget.input.value !== "typed text") fail("chat 421 cleared the typed text: " + JSON.stringify(widget.input.value));
  if (widget.input.disabled !== true) fail("chat 421 input stayed enabled");
  if (widget.send.disabled !== true) fail("chat 421 send stayed enabled");
  if (widget.log.textContent.indexOf("421") !== -1) fail("status number leaked into the panel");
}

function assertChatClosed(page, status) {
  const widget = page.widget;
  if (widget.log.textContent.indexOf(policy.closed) === -1) fail("chat " + status + " lost the closed card");
  if (widget.log.textContent.indexOf(SENTENCE) !== -1) fail("chat " + status + " showed the blocked sentence");
  if (!widget.link()) fail("chat " + status + " lost the request link");
  if (widget.input.value !== "") fail("chat " + status + " changed the cleared input");
  if (widget.input.disabled !== true || widget.send.disabled !== true) fail("chat " + status + " controls stayed enabled");
}

// Wake form (console.html, sleep.js).
const sleepSource = fs.readFileSync(path.join(root, "olympus/sleep.js"), "utf8");
const wakeSource = fs.readFileSync(path.join(root, "olympus/wake-state.js"), "utf8");

function renderWakeForm(status) {
  const doc = makeDocument();
  const ids = {};
  ["sleep-portal", "sleep-panel", "sleep-poster", "sleep-live"].forEach(function (id) {
    const node = new Node(doc, "div");
    node.setAttribute("id", id);
    ids[id] = node;
    doc.body.appendChild(node);
  });
  ids["sleep-portal"].setAttribute("data-endpoint", "/wake");
  doc.getElementById = function (id) { return ids[id] || null; };
  const context = {
    console: console,
    URLSearchParams: URLSearchParams,
    Intl: Intl,
    Date: Date,
    setTimeout: setTimeout,
    document: doc,
    location: { search: "?state=form&where=elsewhere", href: "http://console.local/console.html?state=form" },
    fetch: function (url) {
      if (url !== "/wake") fail("wake request went to " + url);
      return Promise.resolve(response(status));
    },
  };
  context.window = context;
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(wakeSource, context, { filename: "wake-state.js" });
  vm.runInContext(sleepSource, context, { filename: "sleep.js" });
  const form = ids["sleep-panel"].children.filter(function (node) { return node.tag === "form"; })[0];
  if (!form) fail("wake form did not render");
  ["name", "company", "email", "when"].forEach(function (name) {
    form[name] = form.querySelector("input[name=" + name + "]");
    if (!form[name]) fail("wake form is missing " + name);
  });
  form.name.value = "Ada";
  form.company.value = "Analytical Engines";
  form.email.value = "ada@example.com";
  form.querySelector("input[name=where]:checked").checked = true;
  const send = form.querySelector("button[type=submit]");
  const error = form.all().filter(function (node) { return node.className === "sleep-error"; })[0];
  if (!send || !error) fail("wake form is missing the submit button or the error line");
  (form.listeners.submit || []).forEach(function (fn) { fn({ preventDefault: function () {} }); });
  return flush().then(function () { return { form: form, send: send, error: error }; });
}

function assertWakeBlocked(page) {
  const notes = page.form.all().filter(function (node) { return node.getAttribute("role") === "status"; });
  if (notes.length !== 1) fail("wake 421 status slots: " + notes.length);
  if (notes[0].textContent !== SENTENCE) fail("wake 421 sentence was " + JSON.stringify(notes[0].textContent));
  if (notes[0].children.length !== 0) fail("wake 421 status slot has child nodes");
  if (notes[0].className !== "sleep-tiny") fail("wake 421 note class " + notes[0].className);
  if (page.send.parent || page.form.querySelector("button[type=submit]")) fail("wake 421 left the submit button in the form");
  if (page.error.hidden !== true) fail("wake 421 left the failure line visible");
  const text = page.form.textContent;
  if (text.indexOf("did not send") !== -1 || text.indexOf("has not received") !== -1) fail("wake 421 showed the failure copy");
  if (page.form.all().some(function (node) { return node.tag === "a"; })) fail("wake 421 rendered a link");
  if (hasBodyLeak(text)) fail("wake 421 body leaked into the form");
}

function assertWakeFailed(page, status) {
  if (page.error.hidden !== false) fail("wake " + status + " hid the failure line");
  if (page.error.textContent !== "The request did not send. Lev has not received it.") {
    fail("wake " + status + " failure copy changed: " + page.error.textContent);
  }
  if (page.send.parent !== page.form || page.send.hidden !== false || page.send.disabled !== false) fail("wake " + status + " submit button changed");
  if (page.form.textContent.indexOf(SENTENCE) !== -1) fail("wake " + status + " showed the blocked sentence");
}

Promise.resolve()
  .then(function () { return chatSubmit(421, "typed text"); })
  .then(assertChatBlocked)
  .then(function () { return chatSubmit(403, "typed text"); })
  .then(function (page) { assertChatClosed(page, 403); })
  .then(function () { return chatSubmit(404, "typed text"); })
  .then(function (page) { assertChatClosed(page, 404); })
  .then(function () { return renderWakeForm(421); })
  .then(assertWakeBlocked)
  .then(function () { return renderWakeForm(403); })
  .then(function (page) { assertWakeFailed(page, 403); })
  .then(function () { return renderWakeForm(404); })
  .then(function (page) { assertWakeFailed(page, 404); })
  .then(function () { console.log("blocked state ok"); })
  .catch(function (error) {
    fail(error && error.stack ? error.stack : String(error));
  });
