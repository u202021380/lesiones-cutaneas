const MAX_BYTES = 10 * 1024 * 1024;
const NOMBRES = {
  ACK: "Queratosis actínica", BCC: "Carcinoma basocelular", MEL: "Melanoma",
  NEV: "Nevo melanocítico", SCC: "Carcinoma epidermoide", SEK: "Queratosis seborreica",
};
const $ = (id) => document.getElementById(id);
let archivo = null;
let fotoUrl = null;
let mapaUrl = null;
let mostrandoMapa = true;

// HU11: la guía se puede ocultar; la preferencia se recuerda en este navegador
function guardarPreferencia(oculta) {
  try { localStorage.setItem("guiaOculta", oculta ? "1" : "0"); } catch (e) { /* sin almacenamiento */ }
}
function aplicarGuia(oculta) {
  $("guia").hidden = oculta;
  $("mostrar-guia").hidden = !oculta;
}
let guiaOculta = false;
try { guiaOculta = localStorage.getItem("guiaOculta") === "1"; } catch (e) { guiaOculta = false; }
aplicarGuia(guiaOculta);
$("ocultar-guia").onclick = () => { aplicarGuia(true); guardarPreferencia(true); };
$("mostrar-guia").onclick = () => { aplicarGuia(false); guardarPreferencia(false); };

fetch("api/v1/salud").then((r) => r.json()).then((s) => {
  if (s.modelo_simulado) $("banner-simulado").hidden = false;
}).catch(() => {});

function mostrarError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

// HU01: validación rápida en el navegador (la API vuelve a validar todo)
$("archivo").addEventListener("change", (ev) => {
  const f = ev.target.files[0];
  $("resultado").hidden = true;
  mostrarError("");
  archivo = null;
  $("evaluar").disabled = true;
  if (!f) return;
  if (!["image/jpeg", "image/png"].includes(f.type)) {
    mostrarError("Formato no admitido: use JPG o PNG");
    return;
  }
  if (f.size > MAX_BYTES) {
    mostrarError("El archivo supera el tamaño máximo de 10 MB");
    return;
  }
  archivo = f;
  if (fotoUrl) URL.revokeObjectURL(fotoUrl);
  fotoUrl = URL.createObjectURL(f);
  $("vista-previa").src = fotoUrl;
  $("vista-previa").hidden = false;
  $("evaluar").disabled = false;
});

$("evaluar").addEventListener("click", async () => {
  if (!archivo) return;
  $("evaluar").disabled = true;
  $("evaluar").textContent = "Evaluando…";
  mostrarError("");
  try {
    const datos = new FormData();
    datos.append("imagen", archivo);
    const r = await fetch("api/v1/evaluaciones", { method: "POST", body: datos });
    const j = await r.json();
    if (!r.ok) {
      mostrarError(j.detail || "No se pudo evaluar la foto");
      return;
    }
    mostrarResultado(j);
  } catch (e) {
    mostrarError("No se pudo conectar con el servicio. Intente de nuevo.");
  } finally {
    $("evaluar").disabled = false;
    $("evaluar").textContent = "2. Evaluar";
  }
});

function mostrarResultado(j) {
  $("categoria").textContent = j.nombre_categoria;
  $("grupo").textContent = j.grupo;
  $("confianza").textContent = (j.confianza * 100).toFixed(1) + " %";
  const adv = $("advertencia");
  adv.hidden = !j.advertencia_baja_confianza;
  adv.textContent = j.mensaje_advertencia || "";
  const cont = $("probabilidades");
  cont.innerHTML = "";
  Object.entries(j.probabilidades).sort((a, b) => b[1] - a[1]).forEach(([c, p]) => {
    const fila = document.createElement("div");
    fila.className = "barra-fila";
    fila.innerHTML = `<span>${NOMBRES[c] || c}</span><div class="barra-pista"><div class="barra-valor" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${(p * 100).toFixed(1)} %</span>`;
    cont.appendChild(fila);
  });
  mapaUrl = j.mapa_gradcam;
  mostrandoMapa = true;
  $("mapa").src = mapaUrl;
  $("alternar-mapa").textContent = "Ver foto original";
  $("aviso").textContent = j.aviso;
  $("meta").textContent = `Modelo ${j.version_modelo} · ${j.tiempo_procesamiento_ms} ms` + (j.modelo_simulado ? " · MODELO SIMULADO" : "");
  $("resultado").hidden = false;
  $("resultado").scrollIntoView({ behavior: "smooth" });
}

$("alternar-mapa").addEventListener("click", () => {
  mostrandoMapa = !mostrandoMapa;
  $("mapa").src = mostrandoMapa ? mapaUrl : fotoUrl;
  $("alternar-mapa").textContent = mostrandoMapa ? "Ver foto original" : "Ver mapa Grad-CAM";
});
