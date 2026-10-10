const { createHash } = require("crypto");
const { buildRequest } = require("../../../olympus/chat.js");
const policy = require("../../../olympus/chat-policy.json");

const payload = { question: policy.suggested[0], history: [] };

buildRequest(payload).then(function (request) {
  const want = createHash("sha256").update(request.body).digest("hex");
  if (request.headers["x-amz-content-sha256"] !== want) {
    console.error("header does not match body");
    process.exit(1);
  }
});
