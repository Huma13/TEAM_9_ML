const el = id => document.getElementById(id);
let map, generation = 0, mode;
function clearResult() {
  generation++;
  if (map) { map.remove(); map = null; }
  el("result").hidden = true;
}
async function request(endpoint, options) {
  const response = await fetch("/api/" + endpoint, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed (" + response.status + ")");
  return data;
}
async function connect() {
  clearResult();
  const id = generation;
  el("trip").replaceChildren();
  el("trip").disabled = el("run").disabled = true;
  el("mode").textContent = "";
  el("status").textContent = "Connecting to Laptop 1…";
  try {
    const health = await request("health");
    if (!health.ready || !health.model_loaded) throw new Error(health.error || "The server is still loading.");
    const inventory = await request("trips");
    if (id !== generation) return;
    if (!inventory.trips.length) throw new Error("This mode has no available trips.");
    mode = inventory.mode;
    el("mode").textContent = mode + " · " + health.model_name;
    for (const trip of inventory.trips) el("trip").add(new Option(trip, trip));
    el("run").disabled = el("trip").disabled = false;
    el("status").textContent = inventory.trips.length + " trips available";
  } catch (error) { if (id === generation) el("status").textContent = error.message; }
}
function render(data) {
  el("result").hidden = false;
  el("title").textContent = "Trip " + data.trip_id;
  const stats = data.metrics;
  const pct = v => (100*v).toFixed(2) + "%";
  const fields = [
    ["Top-1 accuracy",stats.pointwise_top1_accuracy,pct],["Sequence LCS F1",stats.sequence_lcs_f1,pct],
    ["Route continuity",stats.route_continuity,pct],["Candidate miss rate",stats.candidate_miss_rate,pct],
    ["Route length",stats.predicted_route_length_m,v=>(v/1000).toFixed(2)+" km"],
    ["GPS readings",stats.n_gps_points,v=>v.toLocaleString()],["Server runtime",data.elapsed_time_ms,v=>v.toFixed(0)+" ms"]
  ];
  el("metrics").replaceChildren();
  for (const [label,value,fmt] of fields) {
    const group = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = label; dd.textContent = Number.isFinite(value) ? fmt(value) : "N/A";
    group.append(dt, dd); el("metrics").append(group);
  }
  map = L.map("map",{preferCanvas:true});
  const bounds = L.latLngBounds([]);
  function arcs(rows,style,frame) {
    const lines = rows.map(row => row.coordinates_lon_lat.map(([lon,lat]) => [lat,lon]));
    if (frame) for (const line of lines) for (const p of line) bounds.extend(p);
    return L.polyline(lines,style).addTo(map);
  }
  const roads = arcs(data.road_network_context,{color:"#9ba9a0",weight:2,opacity:.68},false);
  const route = arcs(data.predicted_route_geometry,{color:"#087453",weight:5,opacity:.9},true);
  const gps = L.layerGroup().addTo(map);
  for (const p of data.gps_points) {
    if (!Number.isFinite(p.lat) || !Number.isFinite(p.lon)) continue;
    const xy = [p.lat,p.lon]; bounds.extend(xy);
    L.circleMarker(xy,{radius:2,color:"#1849a5",weight:1,fillOpacity:.55}).addTo(gps);
  }
  L.control.layers(null,{"GPS readings":gps,"Predicted route":route,"Road network":roads},{collapsed:false}).addTo(map);
  L.control.scale({imperial:false}).addTo(map);
  if (bounds.isValid()) map.fitBounds(bounds.pad(.06),{maxZoom:18}); else map.setView([0,0],2);
}
el("refresh").onclick = connect;
el("trip").onchange = () => { clearResult(); el("run").disabled = false; el("status").textContent = "Ready to predict"; };
el("control").onsubmit = async event => {
  event.preventDefault(); clearResult();
  const id = generation, trip = el("trip").value;
  el("run").disabled = true; el("status").textContent = "Reconstructing route…";
  try {
    const data = await request("predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({trip_id:trip})});
    if (id !== generation) return;
    if (data.trip_id !== trip || data.mode !== mode) throw new Error("Server trip or mode changed. Refresh trips and try again.");
    render(data);
    el("status").textContent = statsMessage(data.metrics.evaluation_available);
  } catch (error) { if (id === generation) { el("result").hidden = true; el("status").textContent = error.message; } }
  finally { if (id === generation) el("run").disabled = false; }
};
function statsMessage(scored) { return scored ? "Prediction and server-side scoring complete." : "Prediction complete. Reference scoring is unavailable."; }
connect();
