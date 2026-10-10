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

  getAttribute(key) {
    return Object.prototype.hasOwnProperty.call(this.attrs, key) ? this.attrs[key] : null;
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
if (widget.bubble.hidden !== true) fail("launcher stayed visible while the panel was open");
widget.close();
if (widget.bubble.hidden !== false) fail("launcher stayed hidden after close");
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
if (widget.send.disabled !== true) fail("send stayed enabled");
if (widget.input.getAttribute("tabindex") !== "-1") fail("disabled input stayed focusable");
if (widget.send.getAttribute("tabindex") !== "-1") fail("disabled send stayed focusable");
if (widget.bubble.hidden !== true) fail("launcher stayed visible on the closed card");

const sequence = document();
const cert = "AWS Certified Solutions Architect – Professional (SAP-C02), 2026.";
const live = "EKS orchestration of agent-api, rag-service, mcp-server, and n8n.";
const panel = createWidget(sequence, {
  policy: policy,
  fixture: "switch",
  certLine: cert,
  liveLine: live,
});
const text = panel.log.textContent;
const platformQuestion = text.indexOf(policy.suggested[2]);
const certQuestion = text.indexOf(policy.suggested[1]);
const liveTag = text.indexOf(policy.label_live);
const liveAnswer = text.indexOf(live);
const divider = text.indexOf(policy.switch);
const cvTag = text.indexOf(policy.label_cv);
const certAnswer = text.indexOf(cert);
if (platformQuestion < 0 || certQuestion < 0 || liveTag < 0 || liveAnswer < 0 || divider < 0 || cvTag < 0 || certAnswer < 0) {
  fail("switch fixture missed a part");
}
if (!(platformQuestion < liveTag && liveTag < liveAnswer && liveAnswer < divider && divider < certQuestion && certQuestion < cvTag && cvTag < certAnswer)) {
  fail("switch sequence is not question, tag, live answer, divider, question, tag, CV answer");
}
const answerBlocks = panel.log.children.filter(function (node) {
  return node.attrs.class === "chat-answer";
});
if (answerBlocks.length !== 2) fail("expected two answer blocks");
answerBlocks.forEach(function (block) {
  if (!block.children[0].attrs.class || block.children[0].attrs.class.indexOf("chat-label") !== 0) {
    fail("source tag is not the first child of the answer");
  }
});
if (panel.log.children.filter(function (node) { return node.attrs.class === "chat-question"; }).length !== 2) {
  fail("visitor question bubbles are missing");
}
if (!panel.log.children.some(function (node) { return node.attrs.class === "chat-switch"; })) {
  fail("switch divider is missing");
}
if (text.indexOf("live look") !== -1 || text.indexOf("walkthrough") !== -1) {
  fail("panel answered with the hero geo note");
}

function chatFetch(status) {
  return function (url) {
    if (url !== policy.api_path) fail("request path was " + url + " not " + policy.api_path);
    return Promise.resolve({
      url: url,
      status: status,
      json: function () {
        throw new Error("not json");
      },
    });
  };
}

function askAndCheck(status, expectClosed) {
  const doc = document();
  const widget = createWidget(doc, {
    policy: policy,
    fetchImpl: chatFetch(status),
  });
  return widget.ask(policy.suggested[0]).then(function () {
    const closed = widget.log.textContent.indexOf(policy.closed) !== -1;
    if (closed !== expectClosed) fail("status " + status + " closed=" + closed);
    if (widget.input.disabled !== expectClosed) fail("status " + status + " input disabled mismatch");
    if (widget.send.disabled !== expectClosed) fail("status " + status + " send disabled mismatch");
    if (expectClosed && doc.activeElement !== widget.link()) {
      fail("403 did not focus the request link");
    }
    if (widget.log.textContent.indexOf(String(status)) !== -1) {
      fail("status number leaked into the panel");
    }
  });
}

askAndCheck(500, false).then(function () {
  return askAndCheck(403, true);
}).catch(function (error) {
  fail(error && error.message ? error.message : "waf test failed");
});
