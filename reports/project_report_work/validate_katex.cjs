const fs = require('fs');
const path = require('path');
const katex = require('katex');

const report = process.argv[2];
if (!report) {
  throw new Error('usage: node validate_katex.cjs <report.md>');
}

const original = fs.readFileSync(report, 'utf8');

// Ignore fenced and inline code: delimiter examples and shell snippets are not math.
let text = original.replace(/```[\s\S]*?```/g, '');
text = text.replace(/`[^`\n]*`/g, '');

const displays = [];
text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_match, expression) => {
  displays.push(expression.trim());
  return '';
});

const inlines = [];
text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression) => {
  inlines.push(expression.trim());
  return '';
});

const failures = [];
for (const [kind, expressions, displayMode] of [
  ['display', displays, true],
  ['inline', inlines, false],
]) {
  expressions.forEach((expression, index) => {
    try {
      katex.renderToString(expression, {
        displayMode,
        throwOnError: true,
        strict: 'error',
        output: 'htmlAndMathml',
      });
    } catch (error) {
      failures.push({ kind, index: index + 1, expression, error: String(error) });
    }
  });
}

const delimiterChecks = {
  display_delimiter_count: (text.match(/\$\$/g) || []).length,
  remaining_inline_open: (text.match(/\\\(/g) || []).length,
  remaining_inline_close: (text.match(/\\\)/g) || []).length,
  remaining_single_dollar: (text.match(/(^|[^\\$])\$(?!\$)/g) || []).length,
};

const images = [...original.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)].map((match) => match[1]);
const missingImages = images.filter((candidate) => {
  if (/^https?:\/\//i.test(candidate)) return false;
  return !fs.existsSync(path.normalize(candidate));
});

const localLinks = [...original.matchAll(/(?<!!)\[[^\]]+\]\(([^)]+)\)/g)]
  .map((match) => match[1])
  .filter((candidate) => /^[A-Za-z]:\//.test(candidate));
const missingLinks = localLinks.filter((candidate) => !fs.existsSync(path.normalize(candidate)));

const result = {
  report,
  bytes: Buffer.byteLength(original, 'utf8'),
  lines: original.split(/\r?\n/).length,
  headings: (original.match(/^#{1,6} /gm) || []).length,
  tables: (original.match(/^\|(?:---|:?--)/gm) || []).length,
  formulas: { display: displays.length, inline: inlines.length },
  images: { count: images.length, missing: missingImages },
  local_links: { count: localLinks.length, missing: missingLinks },
  delimiter_checks: delimiterChecks,
  katex_failures: failures,
};

console.log(JSON.stringify(result, null, 2));

if (
  failures.length ||
  missingImages.length ||
  missingLinks.length ||
  delimiterChecks.display_delimiter_count !== 0 ||
  delimiterChecks.remaining_inline_open !== 0 ||
  delimiterChecks.remaining_inline_close !== 0 ||
  delimiterChecks.remaining_single_dollar !== 0
) {
  process.exit(1);
}
