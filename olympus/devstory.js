/** Development story animation. Figures, dates, and labels come from console-data.js. */
(function () {
  "use strict";

  var story = window.OLYMPUS_CONSOLE && window.OLYMPUS_CONSOLE.developmentStory;
  if (!story) return;

  var NS = "http://www.w3.org/2000/svg";
  var TEAL = "#2dd4bf";
  var GREY = "#a3a3a3";
  var GHOST = "#161616";
  var BG = "#0a0a0a";
  var TEXT = "#f5f5f5";
  var BODY = "#e8e8e8";
  var BORD = "#2a2a2a";
  var EDGE_OLD = "#3d3d3d";
  var MONO = "ui-monospace, Consolas, monospace";
  var SANS = '"Segoe UI", system-ui, sans-serif';
  var RAIN = "0123456789<>/{}[]#$%=+*;:";
  var SCENE_MS = 3200;
  var FADE_MS = 400;
  var SCRAMBLE_MS = 520;
  var FILL_MS = 240;
  var STAGGER_MS = 90;
  var EDGE_MS = 480;
  var EDGE_DELAY = 120;
  var CHAR_MS = 38;
  var LABEL_MS = 30;
  var BLINK_MS = 530;
  var RAIN_STEP = 90;
  var RESOLVE_MS = SCRAMBLE_MS + FILL_MS;

  var easeFade = cubicBezier(0.4, 0, 0.2, 1);
  var easeEdge = cubicBezier(0.2, 0.8, 0.2, 1);
  var easeOut = cubicBezier(0, 0, 0.58, 1);
  var nodesById = {};
  story.nodes.forEach(function (node) {
    nodesById[node.id] = node;
  });

  fillPage();
  document.querySelectorAll("[data-devstory]").forEach(mountStage);

  function fillPage() {
    var beats = document.getElementById("ds-beats");
    if (beats) {
      story.beats.forEach(function (beat) {
        var li = document.createElement("li");
        var h = document.createElement("h2");
        h.textContent = beat.title;
        var p = document.createElement("p");
        p.textContent = beat.body;
        li.appendChild(h);
        li.appendChild(p);
        if (beat.cta) {
          var a = document.createElement("a");
          a.className = "btn btn-primary";
          a.href = story.lookHref;
          a.textContent = story.look;
          li.appendChild(a);
        }
        beats.appendChild(li);
      });
    }
    var end = document.getElementById("ds-end");
    if (end) end.appendChild(buildEnd());
  }

  function buildEnd() {
    var frag = document.createDocumentFragment();
    var spend = document.createElement("div");
    spend.className = "ds-spend";
    var kicker = document.createElement("h2");
    kicker.className = "ds-kicker";
    kicker.textContent = story.serviceHeading;
    spend.appendChild(kicker);
    var max = 0;
    story.services.forEach(function (row) {
      if (row.amount > max) max = row.amount;
    });
    var list = document.createElement("ul");
    list.className = "ds-bars";
    story.services.forEach(function (row) {
      var li = document.createElement("li");
      li.className = "ds-row";
      var name = document.createElement("span");
      name.className = "ds-name";
      name.textContent = row.name;
      var meter = document.createElement("span");
      meter.className = "ds-meter";
      var svg = svgEl("svg");
      svg.setAttribute("viewBox", "0 0 100 13");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("aria-hidden", "true");
      var rect = svgEl("rect");
      rect.setAttribute("x", "0");
      rect.setAttribute("y", "0");
      rect.setAttribute("height", "13");
      rect.setAttribute("rx", "2");
      rect.setAttribute("width", String(Math.max(3, (row.amount / max) * 100)));
      if (row.name === "Bedrock") rect.setAttribute("class", "is-teal");
      svg.appendChild(rect);
      meter.appendChild(svg);
      var amt = document.createElement("span");
      amt.className = "ds-amt";
      amt.textContent = "$" + row.amount.toFixed(2);
      li.appendChild(name);
      li.appendChild(meter);
      li.appendChild(amt);
      list.appendChild(li);
    });
    spend.appendChild(list);
    story.endLines.forEach(function (line, index) {
      var p = document.createElement("p");
      p.className = "ds-term";
      var gt = document.createElement("span");
      gt.className = "ds-gt";
      gt.setAttribute("aria-hidden", "true");
      gt.textContent = "> ";
      p.appendChild(gt);
      p.appendChild(document.createTextNode(line));
      if (index === story.endLines.length - 1) {
        var cursor = document.createElement("span");
        cursor.className = "ds-cursor";
        cursor.setAttribute("aria-hidden", "true");
        cursor.textContent = "█";
        p.appendChild(cursor);
      }
      spend.appendChild(p);
    });
    frag.appendChild(spend);

    var blocks = document.createElement("div");
    blocks.className = "ds-blocks";
    blocks.appendChild(block(story.builtHeading, story.builtText));
    blocks.appendChild(block(story.costHeading, story.costText));
    frag.appendChild(blocks);
    return frag;
  }

  function block(title, text) {
    var section = document.createElement("section");
    section.className = "ds-block";
    var h = document.createElement("h2");
    h.textContent = title;
    var p = document.createElement("p");
    p.textContent = text;
    section.appendChild(h);
    section.appendChild(p);
    return section;
  }

  function mountStage(stage) {
    var variant = stage.getAttribute("data-variant") || "hero";
    var reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    var narrowQuery = window.matchMedia("(max-width: 499px)");
    var svg = svgEl("svg");
    svg.setAttribute("class", "ds-svg");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    var rainG = svgEl("g");
    var backG = svgEl("g");
    var frontG = svgEl("g");
    svg.appendChild(rainG);
    svg.appendChild(backG);
    svg.appendChild(frontG);
    stage.appendChild(svg);

    var sr = document.createElement("p");
    sr.className = "visually-hidden";
    stage.appendChild(sr);

    var controls = document.createElement("div");
    controls.className = "ds-controls";
    var pauseBtn = document.createElement("button");
    pauseBtn.type = "button";
    pauseBtn.className = "ds-ctrl";
    pauseBtn.setAttribute("aria-pressed", "false");
    var skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "ds-ctrl";
    skipBtn.textContent = "Skip";
    skipBtn.setAttribute("aria-label", "Skip the animation");
    controls.appendChild(pauseBtn);
    controls.appendChild(skipBtn);
    stage.appendChild(controls);

    var reasons = { hover: false, focus: false, off: true, tab: document.hidden, user: false, reduced: reducedQuery.matches, skip: false };
    var kind = layoutKind();
    var size = layoutSize(kind);
    var rain = null;
    var active = 0;
    var sceneIndex = 0;
    var sceneT = 0;
    var started = false;
    var blinkT = 0;
    var blinkOn = true;
    var raf = 0;
    var last = 0;
    var groups = [backG, frontG];
    var gradSerial = 0;

    applyViewBox();
    rain = buildRain(size.w, size.h, 11);
    drawRain(true);
    if (reasons.reduced) showFinal();
    else renderActive();
    syncMotionClass();
    labelPause();

    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        reasons.off = !entries[0].isIntersecting;
        kick();
      }, { threshold: 0.05 });
      observer.observe(stage);
    } else {
      reasons.off = false;
    }

    document.addEventListener("visibilitychange", function () {
      reasons.tab = document.hidden;
      kick();
    });
    stage.addEventListener("pointerenter", function (event) {
      if (event.pointerType === "touch") return;
      reasons.hover = true;
      kick();
    });
    stage.addEventListener("pointerleave", function (event) {
      if (event.pointerType === "touch") return;
      reasons.hover = false;
      kick();
    });
    stage.addEventListener("focusin", function (event) {
      reasons.focus = true;
      kick();
    });
    stage.addEventListener("focusout", function (event) {
      if (!stage.contains(event.relatedTarget)) reasons.focus = false;
      kick();
    });
    pauseBtn.addEventListener("click", function () {
      if (reasons.skip || reasons.reduced) return;
      reasons.user = !reasons.user;
      if (!reasons.user) reasons.focus = false;
      labelPause();
      kick();
    });
    skipBtn.addEventListener("click", function () {
      reasons.skip = true;
      showFinal();
      labelPause();
      syncMotionClass();
    });
    reducedQuery.addEventListener("change", function () {
      reasons.reduced = reducedQuery.matches;
      if (reasons.reduced) showFinal();
      labelPause();
      syncMotionClass();
      kick();
    });
    if ("ResizeObserver" in window) {
      new ResizeObserver(function () {
        if (reasons.reduced || reasons.skip) showFinal();
        else renderActive();
      }).observe(stage);
    }
    narrowQuery.addEventListener("change", function () {
      var next = layoutKind();
      if (next === kind) return;
      kind = next;
      size = layoutSize(kind);
      applyViewBox();
      rain = buildRain(size.w, size.h, 11);
      drawRain(true);
      if (reasons.reduced || reasons.skip) showFinal();
      else renderActive();
    });
    kick();

    function layoutKind() {
      if (variant === "embed" && !narrowQuery.matches) return "desktop";
      return narrowQuery.matches ? "mobile" : "desktop";
    }

    function layoutSize(next) {
      return next === "mobile" ? { w: 390, h: 844 } : { w: 1280, h: 800 };
    }

    function applyViewBox() {
      svg.setAttribute("viewBox", "0 0 " + size.w + " " + size.h);
    }

    function running() {
      return !reasons.hover && !reasons.focus && !reasons.off && !reasons.tab && !reasons.user && !reasons.reduced && !reasons.skip;
    }

    function labelPause() {
      var stopped = reasons.user || reasons.skip || reasons.reduced;
      pauseBtn.textContent = stopped ? "Play" : "Pause";
      pauseBtn.setAttribute("aria-pressed", stopped ? "true" : "false");
      pauseBtn.setAttribute("aria-label", stopped ? "Play the animation" : "Pause the animation");
      if (reasons.skip || reasons.reduced) {
        pauseBtn.textContent = "Pause";
        pauseBtn.setAttribute("aria-pressed", "true");
        pauseBtn.setAttribute("aria-label", "Animation stopped");
      }
    }

    function syncMotionClass() {
      document.body.classList.toggle("ds-motion", running());
    }

    function kick() {
      syncMotionClass();
      labelPause();
      if (raf) return;
      last = 0;
      raf = window.requestAnimationFrame(tick);
    }

    function tick(now) {
      raf = 0;
      if (!last) last = now;
      var dt = Math.min(64, now - last);
      last = now;
      var play = running();
      if (play) {
        sceneT += dt;
        blinkT += dt;
        blinkOn = Math.floor(blinkT / BLINK_MS) % 2 === 0;
        tickRain(dt);
        if (sceneT >= SCENE_MS) {
          groups[1 - active].setAttribute("opacity", "1");
          active = 1 - active;
          sceneIndex = (sceneIndex + 1) % story.scenes.length;
          sceneT = 0;
          started = true;
          announce(sceneIndex);
        }
        renderActive();
        raf = window.requestAnimationFrame(tick);
      } else {
        blinkOn = reasons.reduced || reasons.skip ? true : blinkOn;
        drawRain(true);
        if (!reasons.reduced && !reasons.skip) renderActive();
      }
      syncMotionClass();
    }

    function showFinal() {
      if (raf) {
        window.cancelAnimationFrame(raf);
        raf = 0;
      }
      active = 0;
      sceneIndex = story.scenes.length - 1;
      sceneT = SCENE_MS;
      started = true;
      blinkOn = true;
      clear(groups[0]);
      clear(groups[1]);
      drawScene(groups[0], story.scenes[sceneIndex], SCENE_MS, true);
      groups[0].setAttribute("opacity", "1");
      groups[1].setAttribute("opacity", "0");
      drawRain(true);
      announce(sceneIndex);
    }

    function renderActive() {
      var scene = story.scenes[sceneIndex];
      drawScene(groups[active], scene, sceneT, false);
      var fade = !started ? 1 : easeFade(clamp(sceneT / FADE_MS));
      groups[active].setAttribute("opacity", fade.toFixed(3));
      groups[1 - active].setAttribute("opacity", (1 - fade).toFixed(3));
      if (sceneT < 48) announce(sceneIndex);
    }

    function announce(index) {
      var scene = story.scenes[index];
      var parts = (scene.captions || []).slice();
      if (scene.proof) parts.push(scene.proof);
      sr.textContent = parts.join(" ");
    }

    function drawScene(group, scene, t, finalFrame) {
      clear(group);
      var mobile = kind === "mobile";
      var m = measures(mobile, scene);
      header(group, scene.date, m);
      graph(group, scene, t, m, finalFrame);
      if (scene.headline) {
        text(group, size.w / 2, m.headlineY, scene.headline, {
          size: m.headSize,
          anchor: "middle",
          fill: TEXT,
          family: SANS,
          weight: "700",
        });
      }
      captions(group, scene, t, m, finalFrame);
      timeline(group, scene, t, m, finalFrame);
    }

    function measures(mobile, scene) {
      var w = size.w;
      var k = (stage.getBoundingClientRect().width || w) / w;
      var embed = variant === "embed" && !mobile;
      var f = function (s, floor) { return mobile ? s : Math.max(s, (floor || 13) / k); };
      var capY = mobile ? 510 : 560;
      if (scene.headline) capY = mobile ? 552 : Math.max(598, 548 + 1.3 * Math.max(22, 14 / k));
      return {
        mobile: mobile,
        embed: embed,
        r: mobile ? 11 : 13,
        label: mobile ? 16 : f(13),
        kicker: mobile ? 16 : f(13),
        tick: f(13),
        caption: mobile ? 16 : f(22, 14),
        captionLh: mobile ? 26 : f(22, 14) * 1.45,
        proof: mobile ? 16 : f(14),
        headSize: mobile ? 26 : f(32, 16),
        dateSize: mobile ? 20 : f(24, 14),
        labelGap: mobile ? 20 : Math.max(20, f(13) + 7),
        graph: mobile ? [34, 110, w - 40, 420] : [40, 110, w - 40, 450],
        capY: capY,
        timelineY: mobile ? 740 : 708,
        timelineX0: mobile ? 24 : 120,
        timelineX1: mobile ? w - 24 : w - 120,
        headlineY: mobile ? 510 : 548,
        barMin: mobile ? 7 : 9,
      };
    }

    function header(group, date, m) {
      if (m.mobile) {
        text(group, 20, 34, story.kicker, { size: m.kicker, fill: GREY, family: SANS, spacing: "2" });
        text(group, 20, 62, date, { size: m.dateSize, fill: TEXT, family: MONO, weight: "700" });
        return;
      }
      text(group, 40, 44, story.kicker, { size: m.kicker, fill: GREY, family: SANS, spacing: "2" });
      text(group, 40, Math.max(76, 44 + m.dateSize * 1.25), date, { size: m.dateSize, fill: TEXT, family: MONO, weight: "700" });
    }

    function stateOf(scene, burst) {
      if (scene.shown.indexOf(burst) === -1) return "ghost";
      if (scene.active.indexOf(burst) !== -1) return "new";
      return "old";
    }

    function decoding(scene, burst) {
      return scene.decode.indexOf(burst) !== -1;
    }

    function staggerOf(node) {
      var order = 0;
      var found = 0;
      story.nodes.forEach(function (item) {
        if (item.burst !== node.burst) return;
        if (item.id === node.id) found = order;
        order += 1;
      });
      return found * STAGGER_MS;
    }

    function resolveAt(scene, node) {
      if (!decoding(scene, node.burst)) return 0;
      return staggerOf(node) + RESOLVE_MS;
    }

    function graph(group, scene, t, m, finalFrame) {
      var box = m.graph;
      story.edges.forEach(function (pair) {
        var a = nodesById[pair[0]];
        var b = nodesById[pair[1]];
        var sa = stateOf(scene, a.burst);
        var sb = stateOf(scene, b.burst);
        if (sa === "ghost" || sb === "ghost") return;
        var pa = mapPoint(a, box);
        var pb = mapPoint(b, box);
        var fresh = decoding(scene, a.burst) || decoding(scene, b.burst);
        var start = Math.max(resolveAt(scene, a), resolveAt(scene, b)) + EDGE_DELAY;
        var progress = 1;
        if (fresh && !finalFrame) progress = easeEdge(clamp((t - start) / EDGE_MS));
        if (progress <= 0) return;
        var mixed = (sa === "old" && sb === "new") || (sa === "new" && sb === "old");
        var color = sa === "new" || sb === "new" ? TEAL : EDGE_OLD;
        var width = sa === "new" || sb === "new" ? 2 : 1.5;
        drawEdge(group, pa, pb, progress, color, width, mixed, sa === "old");
      });
      story.nodes.forEach(function (node) {
        var mode = stateOf(scene, node.burst);
        var point = mapPoint(node, box);
        var local = t - staggerOf(node);
        var drawMode = mode;
        if (decoding(scene, node.burst) && !finalFrame) {
          if (local < 0) drawMode = "ghost";
          else if (local < SCRAMBLE_MS) drawMode = "scramble";
          else if (local < RESOLVE_MS) drawMode = "fill";
          else drawMode = "new";
        }
        drawNode(group, node, point, drawMode, m, local, t);
      });
    }

    function drawEdge(group, a, b, progress, color, width, gradient, oldFirst) {
        var x2 = a[0] + (b[0] - a[0]) * progress;
        var y2 = a[1] + (b[1] - a[1]) * progress;
        var line = svgEl("line");
        line.setAttribute("x1", a[0].toFixed(1));
        line.setAttribute("y1", a[1].toFixed(1));
        line.setAttribute("x2", b[0].toFixed(1));
        line.setAttribute("y2", b[1].toFixed(1));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke-width", String(width));
        if (gradient) {
        var id = "eg" + (gradSerial += 1);
        var defs = svgEl("defs");
        var grad = svgEl("linearGradient");
        grad.setAttribute("id", id);
        grad.setAttribute("gradientUnits", "userSpaceOnUse");
        var from = oldFirst ? a : b;
        var to = oldFirst ? b : a;
        grad.setAttribute("x1", from[0].toFixed(1));
        grad.setAttribute("y1", from[1].toFixed(1));
        grad.setAttribute("x2", to[0].toFixed(1));
        grad.setAttribute("y2", to[1].toFixed(1));
        grad.appendChild(stop("0", GREY));
        grad.appendChild(stop("1", TEAL));
        defs.appendChild(grad);
        group.appendChild(defs);
        line.setAttribute("stroke", "url(#" + id + ")");
      } else {
        line.setAttribute("stroke", color);
      }
      var length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      line.setAttribute("stroke-dasharray", length.toFixed(1));
      line.setAttribute("stroke-dashoffset", (length * (1 - progress)).toFixed(1));
      group.appendChild(line);
      if (progress > 0 && progress < 1) {
        var dot = svgEl("circle");
        dot.setAttribute("cx", x2.toFixed(1));
        dot.setAttribute("cy", y2.toFixed(1));
        dot.setAttribute("r", "4");
        dot.setAttribute("fill", TEXT);
        group.appendChild(dot);
      }
    }

    function fitX(x, m, str) {
      if (m.mobile) return x;
      var half = 0.6 * m.label * String(str).length / 2;
      return Math.min(size.w - 12 - half, Math.max(12 + half, x));
    }

    function drawNode(group, node, point, mode, m, local, t) {
      var x = point[0];
      var y = point[1];
      if (mode === "ghost") {
        circle(group, x, y, m.r, GHOST, null, null);
        return;
      }
      if (mode === "scramble") {
        circle(group, x, y, m.r + 2, BG, TEAL, "2");
        text(group, x, y + 5, glyphAt(t, node.x), { size: m.label, fill: TEAL, family: MONO, anchor: "middle" });
        label(group, node, fitX(x, m, node.label), y + m.r + m.labelGap, TEAL, m.label, local, t, true);
        return;
      }
      if (mode === "fill") {
        var p = easeOut(clamp((local - SCRAMBLE_MS) / FILL_MS));
        circle(group, x, y, m.r + 9, TEAL, null, null, (0.13 * p).toFixed(3));
        circle(group, x, y, m.r, TEAL, null, null, p.toFixed(3));
        label(group, node, fitX(x, m, node.label), y + m.r + m.labelGap, TEAL, m.label, local, t, false);
        return;
      }
      var fill = mode === "new" ? TEAL : GREY;
      if (mode === "new") circle(group, x, y, m.r + 9, TEAL, null, null, "0.13");
      circle(group, x, y, m.r, fill, null, null);
      if (m.embed && mode !== "new") return;
      text(group, fitX(x, m, node.label), y + m.r + m.labelGap, node.label, { size: m.label, fill: fill, family: MONO, anchor: "middle" });
    }

    function label(group, node, x, y, fill, size, local, t, scramble) {
      var shown = Math.max(0, Math.min(node.label.length, Math.floor(local / LABEL_MS)));
      var chars = "";
      for (var i = 0; i < node.label.length; i += 1) {
        if (i < shown) chars += node.label.charAt(i);
        else if (scramble) chars += glyphAt(t, node.x + i + 3);
        else break;
      }
      if (!chars) return;
      text(group, x, y, chars, { size: size, fill: fill, family: MONO, anchor: "middle" });
    }

    function captions(group, scene, t, m, finalFrame) {
      var lines = scene.captions || [];
      var typed = finalFrame ? 10000 : Math.floor(t / CHAR_MS);
      var used = 0;
      var activeLine = 0;
      lines.forEach(function (line, index) {
        if (typed >= used) activeLine = index;
        used += line.length;
      });
      used = 0;
      lines.forEach(function (line, index) {
        var take = Math.max(0, Math.min(line.length, typed - used));
        used += line.length;
        var shown = line.slice(0, take);
        var cursor = index === activeLine && (finalFrame || blinkOn);
        if (take === 0 && index > 0) return;
        var content = "> " + shown;
        var node = text(group, capX(m, lines), m.capY + index * m.captionLh, content, {
          size: m.caption,
          fill: TEXT,
          family: MONO,
          anchor: "start",
        });
        paintPrefix(node);
        if (cursor && (take > 0 || t > 0 || finalFrame)) {
          var mark = svgEl("tspan");
          mark.setAttribute("fill", TEAL);
          mark.textContent = "█";
          node.appendChild(mark);
        }
      });
      if (!scene.proof) return;
      var proofAt = lines.reduce(function (sum, line) { return sum + line.length; }, 0);
      var proofTake = finalFrame ? scene.proof.length : Math.max(0, Math.min(scene.proof.length, typed - proofAt));
      if (proofTake <= 0) return;
      text(group, size.w / 2, m.capY + Math.max(0, lines.length - 1) * m.captionLh + (m.mobile ? 28 : Math.max(30, m.proof * 1.6)), scene.proof.slice(0, proofTake), {
        size: m.proof,
        fill: GREY,
        family: SANS,
        anchor: "middle",
      });
    }

    function capX(m, lines) {
      var longest = 0;
      lines.forEach(function (line) {
        if (line.length > longest) longest = line.length;
      });
      var width = 0.6 * m.caption * (2 + longest);
      return size.w / 2 - width / 2;
    }

    function paintPrefix(node) {
      var raw = node.textContent || "";
      node.textContent = "";
      var pre = svgEl("tspan");
      pre.setAttribute("fill", TEAL);
      pre.textContent = "> ";
      node.appendChild(pre);
      node.appendChild(document.createTextNode(raw.slice(2)));
    }

    function timeline(group, scene, t, m, finalFrame) {
      var y = m.timelineY;
      var x0 = m.timelineX0;
      var x1 = m.timelineX1;
      var axis = svgEl("line");
      axis.setAttribute("x1", String(x0));
      axis.setAttribute("x2", String(x1));
      axis.setAttribute("y1", String(y + 22));
      axis.setAttribute("y2", String(y + 22));
      axis.setAttribute("stroke", BORD);
      group.appendChild(axis);
      story.timeline.years.forEach(function (year) {
        var x = dateX(year.date, x0, x1);
        var tick = svgEl("line");
        tick.setAttribute("x1", x.toFixed(0));
        tick.setAttribute("x2", x.toFixed(0));
        tick.setAttribute("y1", String(y + 18));
        tick.setAttribute("y2", String(y + 28));
        tick.setAttribute("stroke", "#555");
        group.appendChild(tick);
        if (m.embed && String(scene.date).indexOf(year.label) === -1) return;
        text(group, x + 4, y + 30 + (m.mobile ? 14 : m.tick), year.label, { size: m.mobile ? 16 : m.tick, fill: GREY, family: MONO });
      });
      if (scene.branch) {
        var progress = finalFrame ? 1 : easeOut(clamp(t / 600));
        var xs = dateX(story.bursts[0].end, x0, x1);
        var xe = dateX(story.bursts[2].start, x0, x1);
        var id = "tb" + (gradSerial += 1);
        var defs = svgEl("defs");
        var grad = svgEl("linearGradient");
        grad.setAttribute("id", id);
        grad.setAttribute("gradientUnits", "userSpaceOnUse");
        grad.setAttribute("x1", xs.toFixed(0));
        grad.setAttribute("y1", "0");
        grad.setAttribute("x2", xe.toFixed(0));
        grad.setAttribute("y2", "0");
        grad.appendChild(stop("0", GREY));
        grad.appendChild(stop("1", TEAL));
        defs.appendChild(grad);
        group.appendChild(defs);
        var branch = svgEl("line");
        var xEnd = xs + (xe - xs) * progress;
        branch.setAttribute("x1", xs.toFixed(0));
        branch.setAttribute("x2", xEnd.toFixed(0));
        branch.setAttribute("y1", String(y + 8));
        branch.setAttribute("y2", String(y + 8));
        branch.setAttribute("stroke", "url(#" + id + ")");
        branch.setAttribute("stroke-width", "1.5");
        var length = Math.abs(xe - xs);
        branch.setAttribute("stroke-dasharray", length.toFixed(1));
        branch.setAttribute("stroke-dashoffset", (length * (1 - progress)).toFixed(1));
        group.appendChild(branch);
      }
      var activeLabel = -1;
      scene.active.forEach(function (index) {
        if (index > activeLabel) activeLabel = index;
      });
      story.bursts.forEach(function (burst, index) {
        var mode = stateOf(scene, index);
        var xa = dateX(burst.start, x0, x1);
        var w = Math.max(m.barMin, dateX(burst.end, x0, x1) - xa);
        var rect = svgEl("rect");
        rect.setAttribute("x", xa.toFixed(0));
        rect.setAttribute("y", String(y));
        rect.setAttribute("width", w.toFixed(0));
        rect.setAttribute("height", "16");
        rect.setAttribute("rx", "2");
        rect.setAttribute("fill", mode === "new" ? TEAL : mode === "old" ? GREY : GHOST);
        if (mode === "ghost") rect.setAttribute("stroke", BORD);
        group.appendChild(rect);
        if ((m.mobile || m.embed) && index !== activeLabel) return;
        var color = mode === "new" ? TEAL : mode === "old" ? GREY : "#8a8a8a";
        var lx = m.mobile ? xa + w / 2 : xa;
        text(group, lx, y - 8, burst.label, {
          size: m.mobile ? 16 : m.tick,
          fill: color,
          family: MONO,
          anchor: m.mobile ? "middle" : "start",
        });
      });
      if (!scene.marker) return;
      var mx = dateX(story.timeline.wake, x0, x1);
      var marker = svgEl("line");
      marker.setAttribute("x1", mx.toFixed(0));
      marker.setAttribute("x2", mx.toFixed(0));
      marker.setAttribute("y1", String(y - 4));
      marker.setAttribute("y2", String(y + 20));
      marker.setAttribute("stroke", TEXT);
      marker.setAttribute("stroke-width", "2");
      group.appendChild(marker);
      circle(group, mx, y - 6, 4, TEXT, null, null);
      text(group, mx - 8, y + 30 + (m.mobile ? 14 : m.tick), story.timeline.wakeLabel, { size: m.mobile ? 16 : m.tick, fill: TEXT, family: MONO, anchor: "end" });
    }

    function dateX(iso, x0, x1) {
      var start = day(story.timeline.start);
      var end = day(story.timeline.end);
      return x0 + ((day(iso) - start) / (end - start)) * (x1 - x0);
    }

    function mapPoint(node, box) {
      var x = box[0] + ((node.x - 120) / (880 - 120)) * (box[2] - box[0]);
      var y = box[1] + ((node.y - 80) / (470 - 80)) * (box[3] - box[1]);
      return [x, y];
    }

    function buildRain(w, h, seed) {
      var rng = mulberry32(seed);
      var cols = [];
      clear(rainG);
      for (var x = 8; x < w; x += 22) {
        if (rng() < 0.35) continue;
        var n = 6 + Math.floor(rng() * 17);
        var glyphs = [];
        for (var i = 0; i < n; i += 1) {
          var el = svgEl("text");
          el.setAttribute("fill", TEAL);
          el.setAttribute("font-family", MONO);
          el.setAttribute("font-size", "13");
          el.textContent = rainChar(rng);
          rainG.appendChild(el);
          glyphs.push({
            el: el,
            ch: el.textContent,
            wait: 120 + Math.floor(rng() * 121),
            age: Math.floor(rng() * 200),
          });
        }
        cols.push({
          x: x,
          y: Math.floor(rng() * h) - 80,
          n: n,
          glyphs: glyphs,
          step: Math.floor(rng() * RAIN_STEP),
        });
      }
      return cols;
    }

    function tickRain(dt) {
      if (!rain) return;
      rain.forEach(function (col) {
        col.step += dt;
        while (col.step >= RAIN_STEP) {
          col.step -= RAIN_STEP;
          col.y += 17;
          if (col.y > size.h + 20) col.y = -col.n * 17;
        }
        col.glyphs.forEach(function (glyph, index) {
          glyph.age += dt;
          if (glyph.age >= glyph.wait) {
            glyph.age = 0;
            glyph.wait = 120 + Math.floor(Math.random() * 121);
            glyph.ch = RAIN.charAt(Math.floor(Math.random() * RAIN.length));
          }
          placeGlyph(col, glyph, index);
        });
      });
    }

    function drawRain(frozen) {
      if (!rain) return;
      rain.forEach(function (col) {
        col.glyphs.forEach(function (glyph, index) {
          if (!frozen) return;
          placeGlyph(col, glyph, index);
        });
      });
    }

    function placeGlyph(col, glyph, index) {
      var y = col.y + index * 17;
      var head = index === col.n - 1;
      var opacity = head ? 0.32 : 0.08 + 0.17 * (index / col.n);
      if (y < 8 || y > size.h - 4) opacity = 0;
      glyph.el.setAttribute("x", String(col.x));
      glyph.el.setAttribute("y", y.toFixed(0));
      glyph.el.setAttribute("fill-opacity", opacity.toFixed(2));
      glyph.el.textContent = glyph.ch;
    }
  }

  function day(iso) {
    var parts = iso.split("-");
    return Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
  }

  function glyphAt(t, salt) {
    var step = Math.floor(Math.max(0, t) / 40);
    var index = Math.abs((step * 17 + salt) % RAIN.length);
    return RAIN.charAt(index);
  }

  function rainChar(rng) {
    return RAIN.charAt(Math.floor(rng() * RAIN.length));
  }

  function circle(group, x, y, r, fill, stroke, sw, opacity) {
    var node = svgEl("circle");
    node.setAttribute("cx", x.toFixed(1));
    node.setAttribute("cy", y.toFixed(1));
    node.setAttribute("r", String(r));
    node.setAttribute("fill", fill);
    if (stroke) node.setAttribute("stroke", stroke);
    if (sw) node.setAttribute("stroke-width", sw);
    if (opacity != null) node.setAttribute("opacity", String(opacity));
    group.appendChild(node);
    return node;
  }

  function text(group, x, y, value, opts) {
    var node = svgEl("text");
    node.setAttribute("x", x.toFixed(1));
    node.setAttribute("y", y.toFixed(1));
    node.setAttribute("fill", opts.fill);
    node.setAttribute("font-family", opts.family);
    node.setAttribute("font-size", String(opts.size));
    if (opts.anchor) node.setAttribute("text-anchor", opts.anchor);
    if (opts.weight) node.setAttribute("font-weight", opts.weight);
    if (opts.spacing) node.setAttribute("letter-spacing", opts.spacing);
    node.textContent = value;
    group.appendChild(node);
    return node;
  }

  function stop(offset, color) {
    var node = svgEl("stop");
    node.setAttribute("offset", offset);
    node.setAttribute("stop-color", color);
    return node;
  }

  function svgEl(name) {
    return document.createElementNS(NS, name);
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function clamp(value) {
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
  }

  function cubicBezier(x1, y1, x2, y2) {
    function a(v1, v2) { return 1 - 3 * v2 + 3 * v1; }
    function b(v1, v2) { return 3 * v2 - 6 * v1; }
    function c(v1) { return 3 * v1; }
    function sample(t, v1, v2) { return ((a(v1, v2) * t + b(v1, v2)) * t + c(v1)) * t; }
    function slope(t, v1, v2) { return 3 * a(v1, v2) * t * t + 2 * b(v1, v2) * t + c(v1); }
    return function (x) {
      if (x <= 0) return 0;
      if (x >= 1) return 1;
      var t = x;
      for (var i = 0; i < 6; i += 1) {
        var dx = sample(t, x1, x2) - x;
        var s = slope(t, x1, x2);
        if (Math.abs(s) < 1e-6) break;
        t -= dx / s;
      }
      return sample(t, y1, y2);
    };
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
})();
