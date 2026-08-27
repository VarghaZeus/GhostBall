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
    this.hidden = false;
    this.onclick = null;
    this.onchange = null;
    this.oninput = null;
    this.onerror = null;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    // Recorded rather than applied: assertions are about what the panel *said*,
    // and a real class list would mean reimplementing CSS to read it back.
    this.classes = new Set();
    this.listeners = {};
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

  addEventListener(type, handler) {
    // Recorded, not ignored. A control whose handler is never invoked is a
    // control that has never actually been tested -- which is how a broken tab
    // bar passed.
    (this.listeners[type] || (this.listeners[type] = [])).push(handler);
  }

  click() {
    const event = { preventDefault() {} };
    for (const handler of this.listeners.click || []) handler(event);
    // `onclick` as well as `addEventListener`. The panel uses both -- the tab
    // bar listens, most buttons assign -- and a harness that only fired one of
    // them would leave whole controls untestable while looking like it could
    // test them.
    if (typeof this.onclick === "function") this.onclick(event);
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    return allElements().filter((el) => el !== this && isDescendant(this, el) && matches(el, selector));
  }

  matches(selector) {
    return matches(this, selector);
  }
}

/** Every element the document knows about: declared in markup or created since. */
function allElements() {
  return [...elements.values(), ...created];
}

function isDescendant(root, candidate) {
  const stack = [...root.children];
  while (stack.length) {
    const node = stack.pop();
    if (node === candidate) return true;
    stack.push(...node.children);
  }
  return false;
}

/** Supports `tag`, `.class`, `tag.class`, `[attr]`, `tag[attr]` and
 *  `[attr="value"]` -- the shapes the panel actually uses.
 *
 *  Anything else **throws**, rather than quietly matching nothing. A selector
 *  the harness cannot evaluate is precisely the hole the tab bug fell through:
 *  the old stub returned `[]` for everything, so the buggy line ran against an
 *  empty list and the test passed. Failing loudly on an unknown selector means
 *  the harness can only ever be too strict, never silently blind. */
function matches(el, selector) {
  const parsed = /^([a-zA-Z][a-zA-Z0-9]*)?((?:\.[\w-]+)*)(?:\[([\w-]+)(?:="([^"]*)")?\])?$/.exec(
    selector.trim()
  );
  if (!parsed) {
    throw new Error(`panel_harness cannot evaluate the selector "${selector}"`);
  }
  const [, tag, classes, attribute, value] = parsed;
  if (!tag && !classes && !attribute) {
    throw new Error(`panel_harness cannot evaluate the selector "${selector}"`);
  }
  if (tag && el.tagName.toLowerCase() !== tag.toLowerCase()) return false;
  for (const cls of (classes || "").split(".").filter(Boolean)) {
    if (!el.classes.has(cls)) return false;
  }
  if (!attribute) return true;

  const dataKey = attribute.startsWith("data-")
    ? attribute
        .slice(5)
        .replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    : null;
  const actual = dataKey && dataKey in el.dataset ? el.dataset[dataKey] : el.getAttribute(attribute);
  if (actual === null || actual === undefined) return false;
  return value === undefined || String(actual) === value;
}

// Every id that appears in the markup gets an element. getElementById throws for
// anything else, which turns a typo'd id -- the same class of bug as the one
// this harness exists for -- into a hard failure instead of a silent no-op.
const ids = new Set();
for (const match of html.matchAll(/\bid="([^"]+)"/g)) ids.add(match[1]);

const elements = new Map();
const created = [];

// Elements are built from the markup with their real tag and attributes, so a
// selector like `section[data-tab]` means here what it means in a browser. The
// previous version made every element a <div> with no attributes, which is why
// a bug in exactly that selector could not be seen.
const TAG_RE = /<(\w+)([^>]*\bid="([^"]+)"[^>]*)>/g;
for (const match of html.matchAll(TAG_RE)) {
  const [, tag, attrs, id] = match;
  const element = new StubElement(tag, id);
  for (const attr of attrs.matchAll(/([\w-]+)="([^"]*)"/g)) {
    const [, name, value] = attr;
    if (name.startsWith("data-")) {
      element.dataset[name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value;
    } else if (name === "class") {
      for (const cls of value.split(/\s+/)) if (cls) element.classes.add(cls);
      element.className = value;
    } else if (name !== "id") {
      element.attributes[name] = value;
    }
  }
  elements.set(id, element);
}
// Sections without an id still have to be findable by `section[data-tab]`.
const SECTION_RE = /<section class="([^"]*)" data-tab="([^"]+)">\s*<h2>([^<]*)<\/h2>/g;
for (const match of html.matchAll(SECTION_RE)) {
  const [, className, tab, heading] = match;
  const element = new StubElement("section", "");
  element.className = className;
  for (const cls of className.split(/\s+/)) if (cls) element.classes.add(cls);
  element.dataset.tab = tab;
  const title = new StubElement("h2", "");
  title.textContent = heading.replace(/&amp;/g, "&").trim();
  element.appendChild(title);
  created.push(element);
  created.push(title);
}
for (const id of ids) if (!elements.has(id)) elements.set(id, new StubElement("div", id));

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
    const element = new StubElement(tag, "");
    // Tracked, because the tab buttons are created at runtime and the code
    // under test finds them with a selector. A createElement that forgets its
    // output makes every dynamically built control invisible to the tests.
    created.push(element);
    return element;
  },
  querySelector(selector) {
    return document.querySelectorAll(selector)[0] || null;
  },
  querySelectorAll(selector) {
    return allElements().filter((el) => matches(el, selector));
  },
};

// --- the API the panel talks to --------------------------------------------

const requested = [];
//: Non-GET calls, with their bodies. `requested` deliberately stays a flat list
//: of paths -- several tests read it as one joined string -- so writes get their
//: own record rather than a change of shape.
const posted = [];
//: path -> how many times it has been fetched, for `__sequence` entries.
const sequenceCounts = {};

function respond(path) {
  requested.push(path);
  let entry = scenario.responses[path];
  if (entry && entry.__sequence) {
    // Successive calls to one path get successive entries, the last repeating.
    // A reboot needs this and no single response can express it: the Pi
    // answers, then stops answering, then answers again, and what the panel
    // says only makes sense across all three.
    const sequence = entry.__sequence;
    const index = Math.min(sequenceCounts[path] || 0, sequence.length - 1);
    sequenceCounts[path] = (sequenceCounts[path] || 0) + 1;
    entry = sequence[index];
  }
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

async function fetchStub(url, options) {
  const path = String(url).replace(/^\/api/, "").split("?")[0];
  const method = ((options && options.method) || "GET").toUpperCase();
  if (method !== "GET") posted.push({ path, method, body: (options && options.body) || null });
  return respond(path);
}

// --- run --------------------------------------------------------------------

const errors = [];
const consoleErrors = [];
const confirms = [];

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
  // Answered from the scenario, and recorded. The panel's destructive controls
  // are confirm-gated, and a harness with no `confirm` would send them down a
  // ReferenceError instead of through the gate -- which is indistinguishable,
  // from the outside, from a gate that works.
  confirm(message) {
    confirms.push(String(message));
    return Boolean(scenario.confirm);
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
  // Real enough to exercise the code paths that use them. A stub that threw --
  // or that was simply absent -- would push the panel down its catch branches,
  // and a test that only ever runs the fallback has not tested the feature.
  location: { hash: scenario.hash || "" },
  history: {
    replaceState(_state, _title, url) {
      context.location.hash = String(url || "").replace(/^.*#/, "#");
    },
  },
  localStorage: (() => {
    const store = new Map(Object.entries(scenario.storage || {}));
    return {
      getItem: (key) => (store.has(key) ? store.get(key) : null),
      setItem: (key, value) => store.set(key, String(value)),
      removeItem: (key) => store.delete(key),
      dump: () => Object.fromEntries(store),
    };
  })(),
  addEventListener() {},
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
const $ = (id) => elements.get(id);

// Wrapped so a throw in the summary produces a diagnosis rather than silence.
// A harness that exits 0 with no output is indistinguishable from one that ran
// nothing, and that ambiguity costs more time than any bug it hides.
process.on("uncaughtException", (err) => {
  process.stdout.write(JSON.stringify({ errors: ["harness crashed: " + (err.stack || err)] }));
  process.exit(1);
});

/** Let every pending promise chain settle before reading the DOM. */
async function settle() {
  for (let i = 0; i < 20; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

(async () => {
  await settle();

  // Then click whatever the scenario asked for, and let *those* handlers settle
  // too. Without this a control could only ever be tested for existing, and
  // "the button is in the markup" is not the same claim as "tapping it does the
  // thing" -- which is the whole reason this harness exists.
  for (const id of scenario.click || []) {
    if (!elements.has(id)) {
      throw new Error(`scenario clicks "${id}", which is not an id in index.html`);
    }
    elements.get(id).click();
  }
  if ((scenario.click || []).length) await settle();

  // Extra status polls, for behaviour that only exists across several of them.
  // The panel's own intervals are stubbed out -- a harness that let them fire
  // would never exit -- so a test needing a second poll has to ask for one.
  for (let i = 0; i < (scenario.pollAgain || 0); i += 1) {
    if (typeof context.poll !== "function") {
      throw new Error("scenario asks for extra polls, but poll() is not a global");
    }
    await context.poll();
    await settle();
  }

  const texts = {};
  const classes = {};
  const shown = {};
  const hidden = {};
  const disabled = {};
  for (const [id, element] of elements) {
    texts[id] = element.textContent;
    classes[id] = element.className;
    shown[id] = element.classes.has("show");
    hidden[id] = Boolean(element.hidden);
    disabled[id] = Boolean(element.disabled);
  }

  // What the tab bar actually ended up as. The bug was three of these being
  // hidden, so it is reported rather than inferred.
  const tabs = ($("tabs") ? $("tabs").children : []).map((button) => ({
    tab: button.dataset.tab,
    label: button.textContent,
    hidden: Boolean(button.hidden),
    selected: button.getAttribute("aria-selected") === "true",
    alert: button.getAttribute("data-alert") === "true",
  }));

  // And which cards are on screen, by tab.
  const sections = allElements()
    .filter((el) => el.tagName === "section" && el.dataset.tab)
    .map((el) => ({
      tab: el.dataset.tab,
      title: (el.querySelector("h2") || {}).textContent || "",
      hidden: Boolean(el.hidden),
      collapsed: el.classes.has("collapsed"),
    }));

  process.stdout.write(
    JSON.stringify(
      { texts, classes, shown, hidden, tabs, sections, requested, posted, confirms,
        errors, consoleErrors, disabled,
        storage: context.localStorage.dump(), hash: context.location.hash },
      null,
      1
    )
  );
})().catch((err) => {
  process.stdout.write(
    JSON.stringify({ errors: ["harness summary failed: " + (err.stack || err)] })
  );
  process.exit(1);
});
