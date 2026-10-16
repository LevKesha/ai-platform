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
    const href = node.href || "";
    return text === "Sign in" || text === "admin login" || LOGIN_HREFS.indexOf(href) >= 0;
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
  ["Live dashboards", "dashboards"],
];
const LOGIN_HREFS = ["/litellm", "https://n8n.levkesha.com", "/cv-jobs/", "/headroom"];
const LOOK_SECTIONS = ["AI spend control", "CV×Jobs", "n8n", "LiteLLM"];
const TILE_LOGINS = [
  ["LiteLLM", "/litellm", "admin login"],
  ["n8n", "https://n8n.levkesha.com", "admin login"],
  ["CV×Jobs", "/cv-jobs/", "Sign in"],
  ["Headroom", "/headroom", "Sign in"],
];

function openSection(doc, label) {
  const button = doc.querySelectorAll("button").filter(function (node) {
    return node.textContent === label;
  })[0];
  if (!button) fail("missing nav " + label);
  button.click();
}

function renderPage(fetchImpl, options) {
  options = options || {};
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
    if (options.skipWake && name === "wake-state.js") return;
    const source = fs.readFileSync(path.join(root, "olympus", name), "utf8");
    vm.runInContext(source, context, { filename: name });
  });
  if (!options.skipWake && !context.olympusWakeState) fail("page scripts did not install wake state");
  if (options.skipWake && context.olympusWakeState) fail("wake state was present");
  if (!context.OLYMPUS_CONSOLE) fail("page scripts did not install console data");
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
    if (mode.kind === "badjson") {
      return Promise.resolve({
        ok: true,
        json: function () { return Promise.reject(new Error("not json")); },
      });
    }
    return Promise.resolve({
      ok: true,
      json: function () { return Promise.resolve(mode.body); },
    });
  };
}

function assertSectionLooks(page, label) {
  LOOK_SECTIONS.forEach(function (name) {
    openSection(page.document, name);
    const looks = page.panel.querySelectorAll("a").filter(function (node) {
      return node.textContent.trim() === "Request a look";
    });
    if (looks.length !== 1 || looks[0].href !== "console.html?state=form" || looks[0].className !== "btn btn-primary") {
      fail(label + " " + name + " Request a look " + looks.map(function (node) {
        return node.className + " " + node.href;
      }).join(","));
    }
    const found = loginControls(page.panel);
    if (found.length) fail(label + " " + name + " showed " + found.map(function (node) { return node.textContent.trim(); }).join(", "));
  });
}

function assertClosed(page, label) {
  if (!page.document.querySelectorAll("button").some(function (node) { return node.textContent === "Live dashboards"; })) {
    fail(label + " missing Live dashboards nav");
  }
  SECTIONS.forEach(function (section) {
    openSection(page.document, section[0]);
    const found = loginControls(page.column).map(function (node) { return node.textContent.trim(); });
    if (found.length) fail(label + " " + section[0] + " showed " + found.join(", "));
  });
  assertSectionLooks(page, label);
  openSection(page.document, "Live dashboards");
  const looks = page.panel.querySelectorAll("a").filter(function (node) {
    return node.textContent.trim() === "Request a look";
  });
  if (looks.length !== 1 || looks[0].href !== "console.html?state=form" || looks[0].className !== "") {
    fail(label + " dashboards Request a look " + looks.map(function (node) {
      return node.className + " " + node.href;
    }).join(","));
  }
  openSection(page.document, "CV×Jobs");
  if (page.panel.textContent.indexOf("Request a walkthrough") < 0) fail(label + " dropped Request a walkthrough");
  const walk = page.panel.querySelectorAll("a").filter(function (node) { return node.textContent.trim() === "Request a walkthrough"; })[0];
  if (!walk || walk.className !== "btn" || walk.href !== "console.html?state=form") fail(label + " changed Request a walkthrough");
}

function assertOpen(page) {
  if (!page.document.querySelectorAll("button").some(function (node) { return node.textContent === "Live dashboards"; })) {
    fail("live missing Live dashboards nav");
  }
  openSection(page.document, "Live dashboards");
  const tiles = page.panel.querySelectorAll(".dash-tile");
  if (tiles.length !== 4) fail("expected 4 tiles, saw " + tiles.length);
  const logins = tiles.map(function (tile) {
    const buttons = tile.querySelectorAll("a");
    if (buttons.length !== 1) {
      fail("tile login count " + buttons.map(function (node) { return node.textContent; }).join(","));
    }
    const name = tile.querySelector(".dash-name");
    return [name ? name.textContent.trim() : "", buttons[0].href, buttons[0].textContent.trim()];
  });
  TILE_LOGINS.forEach(function (expected) {
    const found = logins.some(function (row) {
      return row[0] === expected[0] && row[1] === expected[1] && row[2] === expected[2];
    });
    if (!found) fail("tile set " + logins.map(function (row) { return row.join(" "); }).join(" | "));
  });
  assertSectionLooks(page, "live");
  openSection(page.document, "CV×Jobs");
  const walk = page.panel.querySelectorAll("a").filter(function (node) { return node.textContent.trim() === "Request a walkthrough"; })[0];
  if (!walk || walk.className !== "btn" || walk.href !== "console.html?state=form") fail("live changed Request a walkthrough");
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
  .then(function () { return renderPage(fetchFor({ kind: "json", body: "" })); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "empty body"); }); })
  .then(function () { return renderPage(fetchFor({ kind: "badjson" })); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "non-json"); }); })
  .then(function () { return renderPage(fetchFor({ kind: "json", body: { state: "live" } }), { skipWake: true }); })
  .then(function (page) { return flush().then(function () { assertClosed(page, "wake script absent"); }); })
  .catch(function (error) {
    fail(error && error.stack ? error.stack : String(error));
  });
