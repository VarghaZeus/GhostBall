// Runs the control panel's script against a stub DOM and a stub fetch, then
// prints what ended up on screen as JSON.
//
// This exists because `node --check` is not a test. It proves the file parses,
// which is exactly what a dangling identifier does too -- `paintStatus is not
// defined` is a runtime ReferenceError, and it shipped a fully broken panel
// past a green suite. Anything that claims to test this file has to *execute*
// it.
//
// Deliberately not jsdom. The panel is one dependency-free HTML file served off
// a Pi that may have no internet, and a test that needs `npm install` would not
// run in the place it matters. The stub below is ~150 lines and covers the
// handful of DOM calls the panel actually makes.
//
// Usage:  node panel_harness.js <index.html> <scenario.json>
//
// The scenario file gives the responses each /api path should produce; a value
// of {"__throw": "message"} makes that endpoint reject, which is how a network
// failure is simulated.

"use strict";

const fs = require("fs");
const vm = require("vm");

const [htmlPath, scenarioPath] = process.argv.slice(2);
const html = fs.readFileSync(htmlPath, "utf8");
const scenario = JSON.parse(fs.readFileSync(scenarioPath, "utf8"));

// --- the DOM the panel expects ---------------------------------------------

class StubElement {
  constructor(tag, id) {
    this.tagName = tag;
    this.id = id || "";
    this._text = "";
    this.className = "";
    this.value = "";
    this.src = "";
    this.innerHTML = "";
    this.disabled = false;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    // Recorded rather than applied: assertions are about what the panel *said*,
    // and a real class list would mean reimplementing CSS to read it back.
    this.classes = new Set();
    this.classList = {
      toggle: (name, on) => {
        if (on === undefined) {
          this.classes.has(name) ? this.classes.delete(name) : this.classes.add(name);
        } else if (on) {
          this.classes.add(name);
        } else {
          this.classes.delete(name);
        }
      },
      add: (name) => this.classes.add(name),
      remove: (name) => this.classes.delete(name),
      contains: (name) => this.classes.has(name),
    };
  }

  get textContent() {
    return this._text;
  }

  set textContent(value) {
    this._text = String(value);
    // Assigning textContent clears children in a real DOM; the panel relies on
    // that nowhere, but pretending otherwise would let a bug hide.
    if (this._text !== "") this.children = [];
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return name in this.attributes ? this.attributes[name] : null;
  }

  addEventListener() {}
}

// Every id that appears in the markup gets an element. getElementById throws for
// anything else, which turns a typo'd id -- the same class of bug as the one
// this harness exists for -- into a hard failure instead of a silent no-op.
const ids = new Set();
for (const match of html.matchAll(/\bid="([^"]+)"/g)) ids.add(match[1]);

const elements = new Map();
for (const id of ids) elements.set(id, new StubElement("div", id));

// Seed each element with the placeholder text the markup ships -- the em-dashes
// the panel shows before its first poll. Without this every field starts empty,
// and "empty" is indistinguishable from "painted with an empty string", which
// is precisely the distinction a test for "did anything get painted" rests on.
const ENTITIES = { "&mdash;": "\u2014", "&hellip;": "\u2026", "&middot;": "\u00b7", "&amp;": "&" };
for (const match of html.matchAll(/\bid="([^"]+)"[^>]*>([^<]*)</g)) {
  const [, id, raw] = match;
  const text = raw.replace(/&[a-z]+;/g, (e) => (e in ENTITIES ? ENTITIES[e] : e)).trim();
  if (text && elements.has(id)) elements.get(id)._text = text;
}

const document = {
  getElementById(id) {
    if (!elements.has(id)) {
      throw new Error(`getElementById("${id}"): no such element in index.html`);
    }
    return elements.get(id);
  },
  createElement(tag) {
    return new StubElement(tag, "");
  },
  // The panel only queries [data-drill] and [data-nudge], neither of which
  // exists as an id. Returning nothing is honest: those controls are not under
  // test here, and inventing them would test the harness.
  querySelector() {
    return null;
  },
  querySelectorAll() {
    return [];
  },
};

// --- the API the panel talks to --------------------------------------------

const requested = [];

function respond(path) {
  requested.push(path);
  const entry = scenario.responses[path];
  if (entry === undefined) {
    return { ok: false, status: 404, statusText: "Not Found", json: async () => ({}) };
  }
  if (entry && entry.__throw) {
    // A rejected fetch, i.e. the Pi is unreachable -- not an HTTP error status.
    return Promise.reject(new TypeError(entry.__throw));
  }
  if (entry && entry.__status) {
    return {
      ok: false,
      status: entry.__status,
      statusText: entry.__statusText || "",
      json: async () => entry.body || {},
    };
  }
  return { ok: true, status: 200, statusText: "OK", json: async () => entry };
}

async function fetchStub(url) {
  return respond(String(url).replace(/^\/api/, "").split("?")[0]);
}

// --- run --------------------------------------------------------------------

const errors = [];
const consoleErrors = [];

const context = {
  document,
  fetch: fetchStub,
  console: {
    log() {},
    warn() {},
    error(...parts) {
      consoleErrors.push(parts.map(String).join(" "));
    },
  },
  // Timers are stubbed out: the panel installs three intervals at load, and a
  // harness that let them fire would never exit.
  setInterval() {
    return 0;
  },
  setTimeout(fn, ms) {
    return setTimeout(fn, ms);
  },
  clearInterval() {},
  Date,
  Math,
  JSON,
  Promise,
  Error,
  TypeError,
  URL,
};
context.window = context;
context.globalThis = context;

const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

process.on("unhandledRejection", (err) => errors.push("unhandledRejection: " + err));

try {
  vm.createContext(context);
  new vm.Script(script, { filename: "index.html" }).runInContext(context);
} catch (err) {
  errors.push("script threw at load: " + (err && err.stack ? err.stack : err));
}

// The panel's top-level poll() is not awaited, so drain the microtask queue a
// few times to let its promise chain settle before reading the DOM.
(async () => {
  for (let i = 0; i < 20; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }

  const texts = {};
  const classes = {};
  const shown = {};
  for (const [id, element] of elements) {
    texts[id] = element.textContent;
    classes[id] = element.className;
    shown[id] = element.classes.has("show");
  }

  process.stdout.write(
    JSON.stringify({ texts, classes, shown, requested, errors, consoleErrors }, null, 1)
  );
})();
