(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.olympusWakeState = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var inflight = null;

  function asleep() {
    return { state: "asleep", live: false, approved: false, wakeAt: null };
  }

  function normalize(body) {
    var state = body && body.state;
    if (
      state !== "live" &&
      state !== "waking" &&
      state !== "requested" &&
      state !== "scheduled" &&
      state !== "asleep"
    ) {
      return asleep();
    }
    return {
      state: state,
      live: state === "live",
      approved: !!(body && body.approved === true),
      wakeAt: body && Object.prototype.hasOwnProperty.call(body, "wakeAt") ? body.wakeAt : null,
    };
  }

  function readWakeState(fetchImpl) {
    if (!fetchImpl && inflight) return inflight;
    var fetchFn = fetchImpl || (typeof fetch === "function" ? fetch : null);
    if (!fetchFn) return Promise.resolve(asleep());
    var promise = Promise.resolve()
      .then(function () {
        return fetchFn("/wake-status", { cache: "no-store" });
      })
      .then(function (res) {
        if (!res || !res.ok) return asleep();
        return Promise.resolve(res.json()).then(normalize, asleep);
      })
      .catch(asleep);
    if (!fetchImpl) inflight = promise;
    return promise;
  }

  return { readWakeState: readWakeState };
});
