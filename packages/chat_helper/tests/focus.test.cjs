const { createWidget } = require("../../../olympus/chat.js");
const policy = require("../../../olympus/chat-policy.json");

class Node {
  constructor(doc, tag) {
    this.doc = doc;
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this._text = "";
  }

  setAttribute(key, value) {
    this.attrs[key] = value;
  }

  get textContent() {
    return this._text + this.children.map(function (child) {
      return child.textContent || "";
    }).join("");
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  appendChild(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }

  focus() {
    this.doc.activeElement = this;
  }

  addEventListener(type, fn) {
    this.listeners[type] = this.listeners[type] || [];
    this.listeners[type].push(fn);
  }

  remove() {
    if (!this.parent) return;
    this.parent.children = this.parent.children.filter(function (child) {
      return child !== this;
    }, this);
  }
}

function document() {
  const doc = {
    activeElement: null,
    listeners: {},
    body: null,
  };
  doc.createElement = function (tag) {
    return new Node(doc, tag);
  };
  doc.createTextNode = function (text) {
    const node = new Node(doc, "#text");
    node.textContent = text;
    return node;
  };
  doc.addEventListener = function (type, fn) {
    doc.listeners[type] = doc.listeners[type] || [];
    doc.listeners[type].push(fn);
  };
  doc.body = new Node(doc, "body");
  return doc;
}

function fail(message) {
  console.error(message);
  process.exit(1);
}

const doc = document();
const widget = createWidget(doc, { policy: policy, fixture: "" });
widget.open();
if (doc.activeElement !== widget.closer) fail("open did not focus the panel");
widget.close();
if (doc.activeElement !== widget.bubble) fail("close did not return focus to the bubble");
widget.open();
(doc.listeners.keydown || []).forEach(function (fn) {
  fn({ key: "Escape" });
});
if (doc.activeElement !== widget.bubble) fail("escape did not close the panel");
widget.showClosed();
if (doc.activeElement !== widget.link()) fail("closed card did not focus the request link");
if (!widget.log.textContent.includes(policy.closed)) fail("closed card was not announced");
if (widget.input.disabled !== true) fail("input stayed enabled");
