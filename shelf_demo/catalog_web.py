"""Frontend assets for the Catalog tab (custom HTML table).

Renders the registered SKUs with their uploaded reference photos and a
per-row Delete button in the rightmost column, via two JSON routes
(/catalog/api/list, /catalog/api/delete) plus a static mount for the
reference crops. Same gr.HTML pattern as the Live tab — the template
strings must not contain the $(brace) sequence (gradio templating), so all
markup is built with DOM APIs / string concatenation.
"""
from __future__ import annotations

CATALOG_HTML = """
<div class="sct" data-role="sct-root">
  <div class="sct-bar">
    <button class="sct-btn" data-role="refresh">&#8635; Refresh</button>
    <span class="sct-meta" data-role="meta">Loading...</span>
  </div>
  <table class="sct-table">
    <thead>
      <tr>
        <th>ID</th><th>Name</th><th>Category</th><th>Barcode</th>
        <th>Price</th><th>Refs</th><th>Photos</th><th></th>
      </tr>
    </thead>
    <tbody data-role="tbody">
      <tr><td colspan="8" class="sct-empty">Loading...</td></tr>
    </tbody>
  </table>
  <div class="sct-msg" data-role="msg"></div>
</div>
"""

CATALOG_CSS = """
.sct { width: 100%; font-family: inherit; }
.sct-bar { display: flex; align-items: center; gap: 12px; margin: 4px 0 12px; }
.sct-btn {
  padding: 7px 16px; border-radius: 8px; border: 1px solid #334155;
  background: #1e293b; color: #e2e8f0; font-size: 13.5px; font-weight: 600;
  cursor: pointer;
}
.sct-btn:hover { filter: brightness(1.15); }
.sct-btn:disabled { opacity: 0.45; cursor: default; }
.sct-meta { color: #94a3b8; font-size: 13px; }
.sct-table {
  width: 100%; border-collapse: collapse; font-size: 13.5px;
  background: #0f172a; border-radius: 10px; overflow: hidden;
}
.sct-table th, .sct-table td {
  padding: 9px 12px; text-align: left; border-bottom: 1px solid #1e293b;
  vertical-align: middle;
}
.sct-table th {
  background: #1e293b; color: #94a3b8; font-weight: 600; font-size: 12.5px;
  letter-spacing: 0.03em; white-space: nowrap;
}
.sct-table td { color: #e2e8f0; }
.sct-table tr:last-child td { border-bottom: none; }
.sct-table tr:hover td { background: #16203a; }
.sct-photos { display: flex; gap: 6px; flex-wrap: wrap; max-width: 380px; }
.sct-photos img {
  width: 52px; height: 52px; object-fit: cover; border-radius: 6px;
  border: 1px solid #334155; cursor: zoom-in; background: #1e293b;
}
.sct-nophoto { color: #475569; font-size: 12.5px; }
.sct-del {
  padding: 5px 12px; border-radius: 7px; border: 1px solid #7f1d1d;
  background: #991b1b; color: #fecaca; cursor: pointer; font-size: 12.5px;
  font-weight: 600; white-space: nowrap;
}
.sct-del:hover { filter: brightness(1.2); }
.sct-del:disabled { opacity: 0.45; cursor: default; }
.sct-empty { text-align: center; color: #64748b; padding: 28px 0 !important; }
.sct-msg { margin-top: 10px; font-size: 13px; color: #94a3b8; min-height: 18px; }
.sct-msg.err { color: #f87171; }
"""


def catalog_js() -> str:
    """Catalog-tab JS. No $(brace) sequences allowed (gradio templating)."""
    return """
(function () {
  var root = element.querySelector('[data-role="sct-root"]');
  if (!root || root.dataset.catInit === "1") return;
  root.dataset.catInit = "1";

  var metaEl = root.querySelector('[data-role="meta"]');
  var tbody = root.querySelector('[data-role="tbody"]');
  var msgEl = root.querySelector('[data-role="msg"]');
  var btnRefresh = root.querySelector('[data-role="refresh"]');
  var API = location.pathname.replace(/\\/$/, "") + "catalog/api";

  function setMsg(text, isErr) {
    msgEl.textContent = text || "";
    msgEl.className = "sct-msg" + (isErr ? " err" : "");
  }

  function cell(parent, text) {
    var td = document.createElement("td");
    td.textContent = text;
    parent.appendChild(td);
    return td;
  }

  function renderRows(skus) {
    tbody.innerHTML = "";
    if (!skus.length) {
      var tr = document.createElement("tr");
      var td = document.createElement("td");
      td.colSpan = 8; td.className = "sct-empty";
      td.textContent = "No products registered yet.";
      tr.appendChild(td); tbody.appendChild(tr);
      return;
    }
    skus.forEach(function (s) {
      var tr = document.createElement("tr");
      cell(tr, String(s.sku_id));
      cell(tr, s.name);
      cell(tr, s.category || "-");
      cell(tr, s.barcode || "-");
      cell(tr, s.price ? String(s.price) : "-");
      cell(tr, String(s.n_refs));

      var tdPics = document.createElement("td");
      if (s.photos && s.photos.length) {
        var wrap = document.createElement("div");
        wrap.className = "sct-photos";
        s.photos.forEach(function (url) {
          var img = document.createElement("img");
          img.src = url; img.loading = "lazy"; img.alt = "ref";
          img.onclick = function () { window.open(url, "_blank"); };
          wrap.appendChild(img);
        });
        tdPics.appendChild(wrap);
      } else {
        var sp = document.createElement("span");
        sp.className = "sct-nophoto";
        sp.textContent = "no photo (registered before photo save)";
        tdPics.appendChild(sp);
      }
      tr.appendChild(tdPics);

      var tdDel = document.createElement("td");
      var btn = document.createElement("button");
      btn.className = "sct-del";
      btn.textContent = "Delete";
      btn.setAttribute("data-role", "del-" + s.sku_id);
      btn.onclick = function () {
        if (!window.confirm("Delete SKU #" + s.sku_id
            + " (\\"" + s.name + "\\")? This removes its vectors and photos.")) return;
        btn.disabled = true; btn.textContent = "Deleting...";
        fetch(API + "/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sku_id: s.sku_id })
        }).then(function (r) { return r.json(); }).then(function (resp) {
          if (resp && resp.ok) {
            setMsg("Deleted SKU #" + s.sku_id + " (\\"" + s.name + "\\").", false);
            load();
          } else {
            setMsg("Delete failed: " + ((resp && resp.error) || "unknown"), true);
            btn.disabled = false; btn.textContent = "Delete";
          }
        }).catch(function (e) {
          setMsg("Delete failed: " + ((e && e.message) || e), true);
          btn.disabled = false; btn.textContent = "Delete";
        });
      };
      tdDel.appendChild(btn);
      tr.appendChild(tdDel);
      tbody.appendChild(tr);
    });
  }

  function load() {
    btnRefresh.disabled = true;
    fetch(API + "/list").then(function (r) { return r.json(); })
      .then(function (resp) {
        btnRefresh.disabled = false;
        if (!resp || !resp.ok) {
          metaEl.textContent = "Failed to load catalog";
          setMsg((resp && resp.error) || "unknown error", true);
          return;
        }
        metaEl.textContent = resp.skus.length + " product(s) · "
          + resp.n_vectors + " vectors";
        renderRows(resp.skus);
      }).catch(function (e) {
        btnRefresh.disabled = false;
        metaEl.textContent = "Load failed";
        setMsg(String((e && e.message) || e), true);
      });
  }

  btnRefresh.onclick = load;
  load();
})();
"""
