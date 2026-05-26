document.addEventListener('DOMContentLoaded', function() {
    const dropdown = document.getElementById('folder-dropdown');
    if (dropdown) {
        dropdown.addEventListener('change', function() {
            const selectedFolder = this.value.replace(/\//g, '---');
            window.location.href = `/refresh/${selectedFolder}`;
        });
    }
});

function copyToClipboard() {
    const button = event.target;
    const parentDiv = button.closest('div');
    const siblings = Array.from(parentDiv.children).filter(el => el.tagName === 'P');
    const text = siblings.map(el => el.innerHTML).join('\n');
    navigator.clipboard.writeText(text).then(function() {
        const originalText = button.textContent;
        button.textContent = '[Copied!]';
        setTimeout(function() {
            button.textContent = originalText;
        }, 1500);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('Failed to copy to clipboard');
    });
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.getElementById('sidebar-toggle-button');
    const isCollapsed = sidebar.classList.toggle('collapsed');
    if (toggleButton) toggleButton.innerHTML = isCollapsed ? '&#9776;' : '&times;';
}

function rerenderMath(element, attempt = 0) {
    if (!element) return;
    if (!window.renderMathInElement) {
        if (attempt < 20) {
            setTimeout(() => rerenderMath(element, attempt + 1), 50);
        }
        return;
    }
    renderMathInElement(element, {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false
    });
}

function protectMathBlocks(text) {
    const blocks = [];
    const protectedText = text.replace(/\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$(?:\\.|[^$\\\n])+\$/g, (match) => {
        const token = `CODEXMATHPLACEHOLDER${blocks.length}X`;
        blocks.push(match);
        return token;
    });
    return { protectedText, blocks };
}

const allowedHtmlTags = new Set([
    'a', 'b', 'blockquote', 'br', 'code', 'del', 'details', 'div', 'em',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 'mark',
    'ol', 'p', 'pre', 's', 'small', 'span', 'strong', 'sub', 'summary',
    'sup', 'table', 'tbody', 'td', 'th', 'thead', 'tr', 'u', 'ul'
]);

function protectHtmlTags(text) {
    const tags = [];
    const protectedText = text.replace(/<\/?([A-Za-z][\w-]*)\b[^>]*>/g, (match, tagName) => {
        if (!allowedHtmlTags.has(String(tagName).toLowerCase())) {
            return match;
        }
        const token = `CODEXHTMLPLACEHOLDER${tags.length}X`;
        tags.push(match);
        return token;
    });
    return { protectedText, tags };
}

function escapePlainText(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function restoreHtmlTags(text, tags) {
    return text.replace(/CODEXHTMLPLACEHOLDER(\d+)X/g, (_, index) => tags[Number(index)] ?? '');
}

function restoreMathPlaceholders(text) {
    return text.replace(
        /CODEXMATHPLACEHOLDER(\d+)X/g,
        (_, index) => `<span class="codex-math-placeholder" data-math-index="${index}"></span>`
    );
}

function normalizeMarkdownSource(text) {
    return text.replace(/```xml\b[^\n\r]*\r?\n([\s\S]*?)\r?\n```/gi, '$1');
}

function renderMarkedText(element, text = null) {
    if (!element) return;
    const source = normalizeMarkdownSource(text ?? element.dataset.markdownSource ?? element.textContent ?? '');
    element.dataset.markdownSource = source;
    if (window.marked && typeof window.marked.parse === 'function') {
        const { protectedText: mathProtected, blocks } = protectMathBlocks(source);
        const { protectedText: htmlProtected, tags } = protectHtmlTags(mathProtected);
        const rendered = window.marked.parse(escapePlainText(htmlProtected), { breaks: true });
        element.innerHTML = restoreMathPlaceholders(restoreHtmlTags(rendered, tags));
        element.querySelectorAll('.codex-math-placeholder').forEach((placeholder) => {
            const index = Number(placeholder.dataset.mathIndex);
            placeholder.replaceWith(document.createTextNode(blocks[index] ?? ''));
        });
    } else {
        element.textContent = source;
    }
    rerenderMath(element);
    if (window.hljs) {
        element.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
    }
}

function renderMarkedElements(root = document) {
    root.querySelectorAll('.marked').forEach((element) => {
        if (element.tagName === 'PRE') return;
        renderMarkedText(element);
    });
}

function loadResponseBox(element) {
    if (!element.open || element.hasAttribute('data-loaded')) return;
    const idd = element.getAttribute('id');
    fetch(`/modelinteraction/${idd}`)
        .then(response => response.text())
        .then(data => {
            const wrapper = document.createElement('div');
            wrapper.className = 'conversation-content';
            wrapper.innerHTML = data;
            element.appendChild(wrapper);
            element.setAttribute('data-loaded', true);
            renderMarkedElements(wrapper);
            if (window.hljs) hljs.highlightAll();
        })
        .catch(error => console.error('Error fetching details:', error));
}

function loadHistoryStep(stepId, targetId) {
    const target = document.getElementById(targetId);
    if (!target) return;
    if (!stepId) {
        target.innerHTML = '';
        return;
    }
    fetch(`/historystep/${stepId}`)
        .then(response => response.text())
        .then(data => {
            target.innerHTML = data;
            renderMarkedElements(target);
            if (window.hljs) hljs.highlightAll();
        })
        .catch(error => {
            console.error('Error fetching step:', error);
            target.innerHTML = '<div class="error">Error loading step</div>';
        });
}

function initializeRunTabs(root = document) {
    root.querySelectorAll('.run-tab').forEach((tabContainer) => {
        const buttons = tabContainer.querySelectorAll('.run-tab-button');
        buttons.forEach((button) => {
            button.addEventListener('click', () => {
                const targetId = button.dataset.target;
                tabContainer.querySelectorAll('.run-tab-button').forEach((btn) => btn.classList.remove('active'));
                button.classList.add('active');
                const contentRoot = tabContainer.parentElement;
                contentRoot.querySelectorAll('.run-tabcontent').forEach((panel) => {
                    panel.style.display = panel.id === targetId ? 'block' : 'none';
                });
            });
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const current = document.querySelector('.sidebar-item.current');
    if (current) {
        current.scrollIntoView({ behavior: 'auto', block: 'center' });
    }
    renderMarkedElements(document);
    initializeRunTabs(document);
    document.querySelectorAll('.response-box-details').forEach(function(element) {
        element.addEventListener('toggle', function(event) {
            loadResponseBox(event.target);
        });
        loadResponseBox(element);
    });
});
