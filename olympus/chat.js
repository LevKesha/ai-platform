(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (typeof document !== "undefined") {
    const script = document.currentScript;
    api.mount(document, script);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function sha256Hex(text) {
    const bytes = new TextEncoder().encode(text);
    return crypto.subtle.digest("SHA-256", bytes).then(function (digest) {
      return Array.from(new Uint8Array(digest))
        .map(function (byte) {
          return byte.toString(16).padStart(2, "0");
        })
        .join("");
    });
  }

  function buildRequest(payload) {
    const body = JSON.stringify(payload);
    return sha256Hex(body).then(function (hash) {
      return {
        body: body,
        headers: {
          "content-type": "application/json",
          "x-amz-content-sha256": hash,
        },
      };
    });
  }

  function el(doc, tag, attrs, text) {
    const node = doc.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        if (attrs[key] != null) node.setAttribute(key, attrs[key]);
      });
    }
    if (text != null) node.textContent = text;
    return node;
  }

  function fixtureName(doc, options) {
    if (options && Object.prototype.hasOwnProperty.call(options, "fixture")) {
      return options.fixture || "";
    }
    const view = doc.defaultView;
    if (!view || !view.location) return "";
    return new URLSearchParams(view.location.search).get("chat") || "";
  }

  function createWidget(doc, options) {
    const policy = options.policy;
    const bubble = el(doc, "button", {
      class: "chat-bubble",
      type: "button",
      "aria-label": policy.bubble_name,
      "aria-expanded": "false",
      "aria-controls": "chat-panel",
    }, policy.bubble_name);
    const panel = el(doc, "section", {
      id: "chat-panel",
      class: "chat-panel",
      role: "dialog",
      "aria-labelledby": "chat-title",
    });
    panel.hidden = true;
    const head = el(doc, "div", { class: "chat-head" });
    head.appendChild(el(doc, "h2", { id: "chat-title" }, policy.bubble_name));
    const closer = el(doc, "button", { class: "chat-close", type: "button", "aria-label": "Close chat" }, "Close");
    head.appendChild(closer);
    const log = el(doc, "div", { class: "chat-log", "aria-live": "polite", "aria-relevant": "additions" });
    const privacy = el(doc, "p", { class: "chat-privacy" }, policy.privacy);
    const form = el(doc, "form", { class: "chat-form" });
    const input = el(doc, "input", {
      type: "text",
      placeholder: policy.placeholder,
      "aria-label": policy.placeholder,
    });
    const send = el(doc, "button", { class: "chat-send", type: "submit" }, "Send");
    form.appendChild(input);
    form.appendChild(send);
    panel.appendChild(head);
    panel.appendChild(log);
    panel.appendChild(privacy);
    panel.appendChild(form);
    doc.body.appendChild(bubble);
    doc.body.appendChild(panel);

    let requestLink = null;

    function open() {
      panel.hidden = false;
      bubble.setAttribute("aria-expanded", "true");
      (requestLink || closer).focus();
    }

    function close() {
      panel.hidden = true;
      bubble.setAttribute("aria-expanded", "false");
      bubble.focus();
    }

    function addNote(text, withLink) {
      const note = el(doc, "p", { class: "chat-note" });
      note.appendChild(doc.createTextNode(text + " "));
      if (withLink) {
        note.appendChild(el(doc, "a", { href: policy.link_href }, policy.link_text));
      }
      log.appendChild(note);
    }

    function addAnswer(text, label, withSwitch) {
      const block = el(doc, "div", { class: "chat-answer" });
      block.appendChild(el(doc, "p", { class: "chat-answer-text" }, text));
      block.appendChild(el(doc, "span", { class: "chat-label" }, label));
      log.appendChild(block);
      if (withSwitch) log.appendChild(el(doc, "p", { class: "chat-switch" }, policy.switch));
    }

    function showIdle() {
      log.appendChild(el(doc, "p", { class: "chat-opener" }, policy.opener));
      const list = el(doc, "ul", { class: "chat-suggestions" });
      policy.suggested.forEach(function (question) {
        const item = el(doc, "li");
        const button = el(doc, "button", { type: "button" }, question);
        button.addEventListener("click", function () {
          ask(question);
        });
        item.appendChild(button);
        list.appendChild(item);
      });
      log.appendChild(list);
    }

    function showTyping() {
      const dots = el(doc, "p", { class: "chat-dots", "aria-label": "Waiting for an answer" });
      dots.appendChild(el(doc, "i"));
      dots.appendChild(el(doc, "i"));
      dots.appendChild(el(doc, "i"));
      log.appendChild(dots);
      return dots;
    }

    function showClosed() {
      input.disabled = true;
      send.disabled = true;
      const card = el(doc, "div", { class: "chat-closed", role: "status" });
      card.appendChild(el(doc, "p", {}, policy.closed));
      requestLink = el(doc, "a", { href: policy.link_href, id: "chat-request" }, policy.link_text);
      card.appendChild(requestLink);
      log.appendChild(card);
      panel.hidden = false;
      bubble.setAttribute("aria-expanded", "true");
      requestLink.focus();
    }

    function paintFixture(name, corpusLine) {
      open();
      if (name === "idle") showIdle();
      if (name === "typing") showTyping();
      if (name === "answer") addAnswer(corpusLine, policy.label_cv, false);
      if (name === "offtopic") addNote(policy.off_topic, true);
      if (name === "gap") addNote(policy.not_in_cv, true);
      if (name === "live") addAnswer(corpusLine, policy.label_live, false);
      if (name === "switch") addAnswer(corpusLine, policy.label_live, true);
      if (name === "limit") showClosed();
    }

    function ask(question) {
      if (input.disabled) return;
      const dots = showTyping();
      const payload = {
        chatId: options.chatId || "browser",
        idempotencyKey: (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()),
        question: question,
        history: options.history || [],
      };
      buildRequest(payload)
        .then(function (request) {
          return (options.fetchImpl || fetch)("/chat", {
            method: "POST",
            headers: request.headers,
            body: request.body,
          });
        })
        .then(function (response) {
          return response.json();
        })
        .then(function (payload) {
          dots.remove();
          if (payload.kind === "closed") showClosed();
          else if (payload.kind === "off_topic" || payload.kind === "not_in_cv") addNote(payload.text, true);
          else if (payload.kind === "answer") addAnswer(payload.text, payload.label, payload.switch);
        })
        .catch(function () {
          dots.remove();
        });
    }

    bubble.addEventListener("click", function () {
      if (panel.hidden) {
        if (!log.firstChild) showIdle();
        open();
      } else {
        close();
      }
    });
    closer.addEventListener("click", close);
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      ask(question);
    });
    doc.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !panel.hidden) close();
    });

    const fixture = fixtureName(doc, options);
    if (fixture) paintFixture(fixture, options.corpusLine || "");
    return {
      bubble: bubble,
      panel: panel,
      closer: closer,
      log: log,
      input: input,
      open: open,
      close: close,
      showClosed: showClosed,
      link: function () {
        return requestLink;
      },
    };
  }

  function mount(doc, script) {
    const policyUrl = new URL("chat-policy.json", script.src);
    const corpusUrl = new URL("chat-corpus.json", script.src);
    return fetch(policyUrl)
      .then(function (response) {
        return response.json();
      })
      .then(function (policy) {
        const name = fixtureName(doc, {});
        if (name !== "answer" && name !== "live" && name !== "switch") {
          return createWidget(doc, { policy: policy, corpusLine: "" });
        }
        return fetch(corpusUrl)
          .then(function (response) {
            return response.json();
          })
          .then(function (corpus) {
            return createWidget(doc, { policy: policy, corpusLine: corpus.lines[0].text });
          });
      });
  }

  return { sha256Hex: sha256Hex, buildRequest: buildRequest, createWidget: createWidget, mount: mount };
});
