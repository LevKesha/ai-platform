"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function fail(message) {
  console.error(message);
  process.exit(1);
}

const root = path.join(__dirname, "../../..");
const html = fs.readFileSync(path.join(root, "olympus/console.html"), "utf8");
const scripts = ["wake-state.js", "sleep.js", "console-data.js", "console.js"];
const loaded = [...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map(function (match) {
  return match[1];
});
if (loaded.join(",") !== scripts.join(",")) {
  fail("console.html scripts " + loaded.join(",") + " !== " + scripts.join(","));
}

const VOID = { area: 1, base: 1, br: 1, col: 1, embed: 1, hr: 1, img: 1, input: 1, link: 1, meta: 1, source: 1, track: 1, wbr: 1 };

function decode(text) {
  return text
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

class Node {
  constructor(tag) {
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.parent = null;
    this.listeners = {};
    this.hidden = false;
    this._class = "";
    this._text = "";
  }
  setAttribute(key, value) {
    this.attrs[key] = String(value);
    if (key === "class") this._class = String(value);
  }
  getAttribute(key) {
    return this.attrs[key] == null ? null : this.attrs[key];
  }
  removeAttribute(key) {
    delete this.attrs[key];
  }
  get className() {
    return this._class;
  }
  set className(value) {
    this._class = String(value);
    this.attrs.class = this._class;
  }
  get id() {
    return this.attrs.id || "";
  }
  set id(value) {
    this.attrs.id = String(value);
  }
  get href() {
    return this.attrs.href || "";
  }
  set href(value) {
    this.attrs.href = String(value);
  }
  get textContent() {
    return this._text + this.children.map(function (child) { return child.textContent || ""; }).join("");
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  set innerHTML(value) {
    this._text = "";
    this.children = [];
    parseFragment(String(value), this);
  }
  get innerHTML() {
    return this._text;
  }
  appendChild(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  addEventListener(type, fn) {
    (this.listeners[type] || (this.listeners[type] = [])).push(fn);
  }
  click() {
    (this.listeners.click || []).forEach(function (fn) { fn({ target: this }); }, this);
  }
  focus() {}
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  querySelectorAll(selector) {
    const out = [];
    walk(this, function (node) {
      if (node !== this && matches(node, selector)) out.push(node);
    }.bind(this));
    return out;
  }
}

function walk(node, visit) {
  (node.children || []).forEach(function (child) {
    visit(child);
    walk(child, visit);
  });
}

function matches(node, selector) {
  if (!node.tag || node.tag === "#text") return false;
  const className = node.className || "";
  const classes = className.split(/\s+/);
  if (selector.charAt(0) === ".") return classes.indexOf(selector.slice(1)) >= 0;
  if (selector.charAt(0) === "#") return node.id === selector.slice(1);
  const dot = selector.indexOf(".");
  if (dot > 0) {
    return node.tag === selector.slice(0, dot) && classes.indexOf(selector.slice(dot + 1)) >= 0;
  }
  return node.tag === selector;
}

function parseAttrs(raw, node) {
  const re = /([^\s=]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\s"']+)))?/g;
  let match;
  while ((match = re.exec(raw))) {
    const value = match[2] != null ? match[2] : match[3] != null ? match[3] : match[4] != null ? match[4] : "";
    node.setAttribute(match[1], decode(value));
  }
}

function parseFragment(htmlText, parent) {
  const stack = [parent];
  const re = /<!--[\s\S]*?-->|<(\/?)([A-Za-z][\w:-]*)([^>]*?)(\/?)>|([^<]+)/g;
  let match;
  while ((match = re.exec(htmlText))) {
    if (match[0].indexOf("<!--") === 0) continue;
    if (match[5] != null) {
      const text = decode(match[5]);
      if (!text) continue;
      const node = new Node("#text");
      node._text = text;
      node.textContent = text;
      stack[stack.length - 1].appendChild(node);
      continue;
    }
    const tag = match[2].toLowerCase();
    if (match[1] === "/") {
      while (stack.length > 1 && stack[stack.length - 1].tag !== tag) stack.pop();
      if (stack.length > 1) stack.pop();
      continue;
    }
    const node = new Node(tag);
    parseAttrs(match[3], node);
    stack[stack.length - 1].appendChild(node);
    if (!match[4] && !VOID[tag]) stack.push(node);
  }
}

function visible(node) {
  let current = node;
  while (current) {
    if (current.hidden) return false;
    if (current.getAttribute && current.getAttribute("hidden") != null) return false;
    if (current.getAttribute && current.getAttribute("aria-hidden") === "true") return false;
    current = current.parent;
  }
  return true;
}

function loginControls(rootNode) {
  return rootNode.querySelectorAll("a").concat(rootNode.querySelectorAll("button")).filter(function (node) {
    if (!visible(node)) return false;
    const text = node.textContent.replace(/\s+/g, " ").trim();
    return text === "Sign in" || text.toLowerCase().indexOf("admin login") >= 0;
  });
}

function flush() {
  return new Promise(function (resolve) {
    setTimeout(resolve, 20);
  });
}

const SECTIONS = [
  ["AI spend control", "spend"],
  ["CV×Jobs", "cvjobs"],
  ["n8n", "n8n"],
  ["LiteLLM", "litellm"],
];

function openSection(doc, label) {
  const button = doc.querySelectorAll("button").filter(function (node) {
    return node.textContent === label;
  })[0];
  if (!button) fail("missing nav " + label);
  button.click();
}

function renderPage(fetchImpl) {
  const document = new Node("#document");
  document.cookie = "";
  document.addEventListener = function (type, fn) {
    (document.listeners[type] || (document.listeners[type] = [])).push(fn);
  };
  parseFragment(html, document);
  document.getElementById = function (id) {
    let found = null;
    walk(document, function (node) {
      if (!found && node.attrs && node.attrs.id === id) found = node;
    });
    return found;
  };
  document.createElement = function (tag) {
    return new Node(String(tag).toLowerCase());
  };
  document.createTextNode = function (text) {
    const node = new Node("#text");
    node.textContent = text;
    return node;
  };
  const context = {
    console: console,
    URLSearchParams: URLSearchParams,
    fetch: fetchImpl,
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    document: document,
    location: { hash: "", search: "", href: "http://console.local/console.html" },
    history: { replaceState: function () {} },
  };
  context.window = context;
  context.globalThis = context;
  vm.createContext(context);
  scripts.forEach(function (name) {
    const source = fs.readFileSync(path.join(root, "olympus", name), "utf8");
    vm.runInContext(source, context, { filename: name });
  });
  if (!context.olympusWakeState || !context.OLYMPUS_CONSOLE) fail("page scripts did not install wake state and console data");
  const panel = document.querySelector("#console-panel");
  const column = document.querySelector(".console-column");
  if (!panel || !column) fail("console.html did not render the panel");
  return { document: document, panel: panel, column: column };
}

function fetchFor(mode) {
  return function (url, options) {
    if (url !== "/wake-status") fail("status url " + url);
    if (!options || options.cache !== "no-store") fail("status read was not cache: no-store");
    if (mode.kind === "error") return Promise.reject(new Error("network"));
    if (mode.kind === "http") return Promise.resolve({ ok: false, json: function () { return Promise.resolve({}); } });
    return Promise.resolve({
      ok: true,
      json: function () { return Promise.resolve(mode.body); },
    });
  };
}

function assertClosed(page, label) {
  SECTIONS.forEach(function (section) {
    openSection(page.document, section[0]);
    const found = loginControls(page.column).map(function (node) { return node.textContent.trim(); });
    if (found.length) fail(label + " " + section[0] + " showed " + found.join(", "));
  });
  openSection(page.document, "CV×Jobs");
  if (page.panel.textContent.indexOf("Request a walkthrough") < 0) fail(label + " dropped Request a walkthrough");
  if (page.panel.querySelector(".anon-preview") || page.panel.textContent.indexOf("Job inbox") >= 0) {
    fail(label + " still shows the CV×Jobs preview");
  }
  openSection(page.document, "n8n");
  if (page.panel.textContent.indexOf("Request a look") < 0) fail(label + " dropped Request a look");
}

function assertOpen(page) {
  openSection(page.document, "AI spend control");
  const spend = loginControls(page.panel);
  if (spend.length !== 1 || spend[0].textContent.trim() !== "Sign in" || spend[0].href !== "/headroom") {
    fail("awake spend login " + spend.map(function (node) { return node.textContent + " " + node.href; }).join(","));
  }
  openSection(page.document, "CV×Jobs");
  const jobs = loginControls(page.panel);
  if (jobs.length !== 1 || jobs[0].href !== "/cv-jobs/") fail("awake cvjobs login missing");
  if (page.panel.textContent.indexOf("Request a walkthrough") < 0) fail("awake dropped Request a walkthrough");
  openSection(page.document, "n8n");
  const n8nPanel = loginControls(page.panel);
  if (n8nPanel.length) fail("n8n section rendered its own login");
  const n8nTile = loginControls(page.column).filter(function (node) { return node.textContent.indexOf("n8n") >= 0; });
  if (n8nTile.length !== 1 || n8nTile[0].href !== "https://n8n.levkesha.com") fail("awake n8n admin login missing");
  openSection(page.document, "LiteLLM");
  const litellm = loginControls(page.panel);
  if (litellm.length !== 1 || litellm[0].textContent.trim() !== "admin login" || litellm[0].href !== "/litellm") {
    fail("awake litellm admin login missing");
  }
}

Promise.resolve()
  .then(function () { return renderPage(fetchFor({ kind: "json", body: { state: "asleep", approved: false } })); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "asleep"); }); })
  .then(function () { return renderPage(fetchFor({ kind: "json", body: { state: "live", approved: true } })); })
  .then(function (page) { return flush().then(function () { assertOpen(page); }); })
  .then(function () { return renderPage(fetchFor({ kind: "json", body: { state: "nope" } })); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "unknown"); }); })
  .then(function () { return renderPage(fetchFor({ kind: "error" })); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "fetch error"); }); })
  .then(function () { return renderPage(fetchFor({ kind: "http" })); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "http error"); }); })
  .catch(function (error) {
    fail(error && error.stack ? error.stack : String(error));
  });
