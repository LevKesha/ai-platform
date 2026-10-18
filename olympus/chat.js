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
    const closer = el(doc, "button", { class: "chat-close", type: "button", "aria-label": "Close chat" }, "×");
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
    panel.appendChild(form);
    panel.appendChild(privacy);
    doc.body.appendChild(bubble);
    doc.body.appendChild(panel);

    let requestLink = null;

    function open() {
      panel.hidden = false;
      bubble.hidden = true;
      bubble.setAttribute("aria-expanded", "true");
      (requestLink || closer).focus();
    }

    function close() {
      panel.hidden = true;
      bubble.hidden = false;
      bubble.setAttribute("aria-expanded", "false");
      bubble.focus();
    }

    function addQuestion(text) {
      if (!text) return;
      log.appendChild(el(doc, "p", { class: "chat-question" }, text));
    }

    function addNote(text, withLink) {
      const note = el(doc, "p", { class: "chat-note" });
      note.appendChild(doc.createTextNode(text));
      if (withLink) {
        note.appendChild(doc.createTextNode(" "));
        note.appendChild(el(doc, "a", { href: policy.link_href }, policy.link_text));
      }
      log.appendChild(note);
    }

    function addDivider() {
      const row = el(doc, "div", { class: "chat-switch", role: "note" });
      row.appendChild(el(doc, "span", { class: "chat-switch-rule" }));
      row.appendChild(el(doc, "span", { class: "chat-switch-text" }, policy.switch));
      row.appendChild(el(doc, "span", { class: "chat-switch-rule" }));
      log.appendChild(row);
    }

    function addAnswer(text, label, withSwitch) {
      const block = el(doc, "div", { class: "chat-answer" });
      const tagClass = label === policy.label_rag ? "chat-label chat-label-rag" : "chat-label";
      block.appendChild(el(doc, "p", { class: tagClass }, label));
      block.appendChild(el(doc, "p", { class: "chat-answer-text" }, text));
      log.appendChild(block);
      if (withSwitch) addDivider();
    }

    function addRetry(text) {
      const block = el(doc, "div", { class: "chat-answer" });
      block.appendChild(el(doc, "p", { class: "chat-answer-text" }, text || policy.retry));
      log.appendChild(block);
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
      input.setAttribute("tabindex", "-1");
      send.setAttribute("tabindex", "-1");
      const card = el(doc, "div", { class: "chat-closed", role: "status" });
      card.appendChild(el(doc, "p", {}, policy.closed));
      requestLink = el(doc, "a", { href: policy.link_href, id: "chat-request" }, policy.link_text);
      card.appendChild(requestLink);
      log.appendChild(card);
      open();
    }

    function paintFixture(name) {
      const certQ = policy.suggested[1];
      const platformQ = policy.suggested[2];
      const cert = options.certLine || "";
      const live = options.liveLine || "";
      open();
      if (name === "idle") showIdle();
      if (name === "typing") {
        addQuestion(certQ);
        showTyping();
      }
      if (name === "answer") {
        addQuestion(certQ);
        addAnswer(cert, policy.label_cv, false);
      }
      if (name === "offtopic") {
        addQuestion("What is the weather in London?");
        addNote(policy.off_topic, true);
      }
      if (name === "gap") {
        addQuestion("What is Lev's salary?");
        addNote(policy.not_in_cv, true);
      }
      if (name === "live") {
        addQuestion(platformQ);
        addAnswer(live, policy.label_rag, false);
      }
      if (name === "switch") {
        addQuestion(platformQ);
        addAnswer(live, policy.label_rag, false);
        addDivider();
        addQuestion(certQ);
        addAnswer(cert, policy.label_cv, false);
      }
      if (name === "retry") {
        addQuestion(platformQ);
        addRetry(policy.retry);
      }
      if (name === "limit") showClosed();
    }

    function chatApiPath() {
      const path = policy.api_path;
      if (typeof path !== "string" || path.charAt(0) !== "/") {
        throw new Error("chat policy api_path is required");
      }
      return path;
    }

    function isChatApi(url) {
      const path = String(url).split("?")[0];
      const api = chatApiPath();
      return path === api || path.indexOf(api + "/") === 0;
    }

    const turns = Array.isArray(options.history) ? options.history.slice() : [];

    function ask(question) {
      if (input.disabled) return Promise.resolve();
      addQuestion(question);
      const dots = showTyping();
      const payload = {
        chatId: options.chatId || "browser",
        idempotencyKey: (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()),
        question: question,
        history: turns.slice(),
      };
      const chatPath = chatApiPath();
      return buildRequest(payload)
        .then(function (request) {
          return (options.fetchImpl || fetch)(chatPath, {
            method: "POST",
            headers: request.headers,
            body: request.body,
          });
        })
        .then(function (response) {
          if (isChatApi(chatPath) && response && (response.status === 403 || response.status === 404)) {
            dots.remove();
            showClosed();
            return null;
          }
          return Promise.resolve()
            .then(function () {
              if (!response || typeof response.json !== "function") throw new Error("not json");
              return response.json();
            })
            .then(function (body) {
              return body;
            }, function () {
              if (isChatApi(chatPath)) {
                dots.remove();
                showClosed();
              }
              return null;
            });
        })
        .then(function (payload) {
          if (!payload) return;
          dots.remove();
          if (payload.kind === "closed") showClosed();
          else if (payload.kind === "off_topic" || payload.kind === "not_in_cv") {
            addNote(payload.text, true);
            turns.push({ role: "user", text: question });
            turns.push({ role: "assistant", text: payload.text });
          } else if (payload.kind === "answer") {
            addAnswer(payload.text, payload.label, payload.switch);
            turns.push({ role: "user", text: question });
            turns.push({ role: "assistant", text: payload.text });
          } else if (payload.kind === "retry") {
            addRetry(payload.text);
          }
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
    if (fixture) paintFixture(fixture);
    const api = {
      bubble: bubble,
      panel: panel,
      closer: closer,
      log: log,
      input: input,
      send: send,
      open: open,
      close: close,
      showClosed: showClosed,
      ask: ask,
      wake: null,
      link: function () {
        return requestLink;
      },
    };
    api.wakeReady = readSharedWake(options).then(function (state) {
      api.wake = state;
      return state;
    });
    return api;
  }

  function readSharedWake(options) {
    const shared = typeof globalThis !== "undefined" ? globalThis.olympusWakeState : null;
    if (!shared || typeof shared.readWakeState !== "function") return Promise.resolve(null);
    if (!options.wakeFetch && typeof fetch !== "function") return Promise.resolve(null);
    return shared.readWakeState(options.wakeFetch).then(
      function (state) {
        return state;
      },
      function () {
        return null;
      }
    );
  }

  function ensureWake(doc, script) {
    if (typeof globalThis !== "undefined" && globalThis.olympusWakeState) {
      return Promise.resolve(globalThis.olympusWakeState);
    }
    if (!doc || !script || !script.src || !doc.createElement || typeof URL !== "function") {
      return Promise.resolve(null);
    }
    return new Promise(function (resolve) {
      const tag = doc.createElement("script");
      tag.src = new URL("wake-state.js", script.src).href;
      tag.onload = function () {
        resolve((typeof globalThis !== "undefined" && globalThis.olympusWakeState) || null);
      };
      tag.onerror = function () {
        resolve(null);
      };
      (doc.head || doc.body).appendChild(tag);
    });
  }

  function mount(doc, script) {
    const policyUrl = new URL("chat-policy.json", script.src);
    const corpusUrl = new URL("chat-corpus.json", script.src);
    return ensureWake(doc, script).then(function () {
      return fetch(policyUrl)
      .then(function (response) {
        return response.json();
      })
      .then(function (policy) {
        const name = fixtureName(doc, {});
        function pick(corpus, section, needle) {
          const found = corpus.lines.filter(function (item) {
            return item.section === section && item.text.indexOf(needle) !== -1;
          })[0];
          return found ? found.text : "";
        }
        if (name !== "answer" && name !== "live" && name !== "switch") {
          return createWidget(doc, { policy: policy });
        }
        return fetch(corpusUrl)
          .then(function (response) {
            return response.json();
          })
          .then(function (corpus) {
            return createWidget(doc, {
              policy: policy,
              certLine: pick(corpus, "certifications", "Solutions Architect"),
              liveLine: pick(corpus, "platform", "EKS orchestration"),
            });
          });
      });
    });
  }

  return { sha256Hex: sha256Hex, buildRequest: buildRequest, createWidget: createWidget, mount: mount };
});
