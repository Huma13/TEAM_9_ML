# Laptop 2 evaluator interface

Use Python 3.9 or later. The client uses the standard library and bundled Leaflet. No package installation, training files, model weights or internet map service are required.

1. Connect both laptops to the same Wi-Fi or hotspot. On Laptop 1 start the backend following the root README. Obtain its current Wi-Fi IPv4 address.
2. On Laptop 2 open a terminal in the folder containing ui/. Run the command below, replacing the example address with Laptop 1's address. On Windows substitute py for python if necessary.

       python ui/app.py --api http://192.168.1.23:8000

3. Open http://127.0.0.1:8501 in a browser on Laptop 2. Select a trip, press Predict route, and test another trip. Use the map layer controls to compare GPS readings, predicted arcs and road geometry. Use Refresh trips after switching modes.
4. Capture the real UI result and record the trip ID and time. Ask Laptop 1 to capture the matching receiving-terminal log. Give both to the report owner. Ctrl+C stops the UI.

The UI forwards health, trip inventory and prediction requests to the supplied backend URL. A missing reference produces N/A for scoring while the route remains available.
