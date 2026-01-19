import pandas as pd
import folium
from folium.plugins import MarkerCluster
import json
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from dash import Dash, dcc, html, Input, Output, State
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

def bin_data_by_time(df, time_column='date_taken_year', freq='YS'):
    """
    Bins the dataframe into time periods for temporal visualization.
    
    Args:
        df: DataFrame with date columns
        time_column: Column to use for binning ('date_taken_year', 'date_taken_month', etc.)
        freq: Frequency for binning ('YS' for year, 'MS' for month, 'D' for day)
    
    Returns:
        dict: Dictionary with timestamps as keys and GeoJSON features as values
    """
    # Create full datetime from separate columns with correct order
    df_copy = df.copy()
    
    df_copy['datetime'] = pd.to_datetime(
        pd.DataFrame({
            'year': df_copy['date_taken_year'],
            'month': df_copy['date_taken_month'],
            'day': df_copy['date_taken_day']
        }),
        errors='coerce'
    )
    
    # Remove rows with invalid dates
    df = df_copy.dropna(subset=['datetime'])
    
    # Bin by time period
    if freq == 'YS':  # Yearly
        df['time_period'] = df['datetime'].dt.to_period('Y')
    elif freq == 'MS':  # Monthly
        df['time_period'] = df['datetime'].dt.to_period('M')
    elif freq == 'D':  # Daily
        df['time_period'] = df['datetime'].dt.to_period('D')
    
    # Group by time period
    time_bins = {}
    for period, group in df.groupby('time_period'):
        # Convert period to timestamp string (ISO format)
        start_date = period.start_time
        timestamp = start_date.isoformat()
        
        # Create GeoJSON features for this time period
        features = []
        for idx, row in group.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['long'], row['lat']]
                },
                "properties": {
                    "id": str(row['id']),
                    "user": str(row['user']),
                    "date": row['datetime'].strftime('%Y-%m-%d') if pd.notna(row['datetime']) else 'Unknown',
                    "text": str(row['text_merged']) if pd.notna(row['text_merged']) else ''
                }
            }
            features.append(feature)
        
        # Create FeatureCollection for this time period
        time_bins[timestamp] = {
            "type": "FeatureCollection",
            "features": features
        }
    
    return time_bins, df

def generate_temporal_lyon_map(csv_path, time_freq='MS', sample_size=None):
    """
    Generates a temporal map for Lyon data with month-based layer control.
    
    Args:
        csv_path: Path to CSV file
        time_freq: Frequency for binning ('YS' for year, 'MS' for month, 'D' for day)
        sample_size: Optional sample size to reduce file size
    """
    # 1. Load data
    df = pd.read_csv(csv_path)
    print(f"Total number of points before cleaning: {len(df)}")
    
    # 2. Basic cleaning
    df = df.drop_duplicates(subset=['id'])
    print(f"Number of points after duplicate removal: {len(df)}")
    
    # Remove missing lat/long
    df = df.dropna(subset=['lat', 'long'])
    print(f"Number of points after removing missing coordinates: {len(df)}")
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} points from {len(df):,} for faster loading...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using {len(df):,} sampled points")
    
    # 3. Bin data by time
    print(f"\nBinning data by {time_freq}...")
    time_bins, df = bin_data_by_time(df, freq=time_freq)
    print(f"Number of time periods: {len(time_bins)}")
    
    # 4. Create the base map
    lyon_map = folium.Map(location=[45.75, 4.85], zoom_start=13)
    
    # 5. Add feature group for each time period
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'gray']
    
    for idx, (timestamp, geojson_data) in enumerate(sorted(time_bins.items())):
        # Create feature group for this time period
        color = colors[idx % len(colors)]
        period_label = timestamp.split('T')[0]  # Just the date part
        
        fg = folium.FeatureGroup(
            name=f'{period_label} ({len(geojson_data["features"])} photos)', 
            show=(idx < 3)  # Show first 3 periods by default
        )
        
        # Add circles for each point
        for feature in geojson_data['features']:
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            
            popup_text = f"""
            <b>Date:</b> {props['date']}<br>
            <b>Photo ID:</b> {props['id']}<br>
            <b>User:</b> {props['user'][:20]}
            """
            
            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=4,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=1
            ).add_to(fg)
        
        fg.add_to(lyon_map)
    
    # 6. Add layer control
    folium.LayerControl(collapsed=True).add_to(lyon_map)
    
    # 7. Save map
    if sample_size:
        output_file = f'../maps/lyon_temporal_map_sample{sample_size}.html'
    else:
        output_file = '../maps/lyon_temporal_map.html'
    
    lyon_map.save(output_file)
    print(f"\n✅ Temporal map generated successfully!")
    print(f"   Points: {len(df):,}")
    print(f"   Time periods: {len(time_bins)}")
    print(f"   File: '{output_file}'")
    print(f"   Use layer control (top right) to toggle time periods on/off")
    
    return time_bins, df

def generate_lyon_map(csv_path):
    """Original function for static map with MarkerCluster."""
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    print(f"Total number of points before cleaning: {len(df)}")
    # 2. Basic cleaning
    df = df.drop_duplicates(subset=['id'])
    print(f"Number of points after duplicate removal: {len(df)}")
    
    # 3. Create map centered on Lyon
    lyon_map = folium.Map(location=[45.75, 4.85], zoom_start=13)
    
    # Use MarkerCluster to avoid browser lag with large datasets
    marker_cluster = MarkerCluster().add_to(lyon_map)

    # Remove rows with missing lat/long
    df = df.dropna(subset=['lat', 'long'])

    # 4. Add points
    for idx, row in df.iterrows():
         folium.Marker(
             location=[row['lat'], row['long']],
             popup=f"Photo ID: {row['id']}",
         ).add_to(marker_cluster)

    # 5. Save
    lyon_map.save('../maps/lyon_tourism_map.html')
    print(f"Map generated with {len(df)} cleaned points. Open 'lyon_tourism_map.html' in your browser.")

def generate_cluster_histogram(cleaned_csv, clustering_csv, output_file='../maps/cluster_timeline.html'):
    """
    Creates an interactive histogram of cluster counts per month.
    
    Args:
        cleaned_csv: Path to cleaned data CSV
        clustering_csv: Path to CSV with cluster assignments
        output_file: Output HTML file path
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not installed. Install with: pip install plotly")
        return
    
    print("Loading data for cluster histogram...")
    
    # Load cleaned data
    df_clean = pd.read_csv(cleaned_csv)
    print(f"Loaded {len(df_clean):,} cleaned records")
    
    # Load clustering results
    try:
        df_clusters = pd.read_csv(clustering_csv)
        print(f"Loaded {len(df_clusters):,} clustered records")
    except FileNotFoundError:
        print(f"❌ Clustering file not found: {clustering_csv}")
        print("   Run main.py first to generate clustering results")
        return
    
    # Merge data
    df_merged = df_clean.merge(
        df_clusters[['id', 'cluster']], 
        on='id', 
        how='inner'
    )
    print(f"Merged {len(df_merged):,} records")
    
    # Create datetime column
    df_merged['datetime'] = pd.to_datetime(
        pd.DataFrame({
            'year': df_merged['date_taken_year'],
            'month': df_merged['date_taken_month'],
            'day': df_merged['date_taken_day']
        }),
        errors='coerce'
    )
    
    df_merged = df_merged.dropna(subset=['datetime'])
    
    # Extract year-month
    df_merged['year_month'] = df_merged['datetime'].dt.to_period('M')
    
    # Count clusters and photos per month
    monthly_stats = []
    for period, group in df_merged.groupby('year_month'):
        n_clusters = group['cluster'].nunique()
        n_photos = len(group)
        n_noise = (group['cluster'] == -1).sum()
        n_clustered = n_photos - n_noise
        
        monthly_stats.append({
            'date': period.start_time,
            'year_month': str(period),
            'n_clusters': n_clusters,
            'n_photos': n_photos,
            'n_clustered': n_clustered,
            'n_noise': n_noise
        })
    
    df_stats = pd.DataFrame(monthly_stats)
    
    print(f"\n📊 Cluster Timeline Statistics:")
    print(f"   Date range: {df_stats['date'].min().date()} to {df_stats['date'].max().date()}")
    print(f"   Total months: {len(df_stats)}")
    print(f"   Avg clusters/month: {df_stats['n_clusters'].mean():.1f}")
    print(f"   Max clusters: {df_stats['n_clusters'].max()}")
    print(f"   Min clusters: {df_stats['n_clusters'].min()}")
    
    # Create interactive histogram
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['year_month'],
        y=df_stats['n_clusters'],
        name='Number of Clusters',
        marker=dict(
            color=df_stats['n_clusters'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Clusters")
        ),
        hovertemplate='<b>%{x}</b><br>' +
                      'Clusters: %{y}<br>' +
                      'Photos: %{customdata[0]}<br>' +
                      'Clustered: %{customdata[1]}<br>' +
                      'Noise: %{customdata[2]}<br>' +
                      '<extra></extra>',
        customdata=df_stats[['n_photos', 'n_clustered', 'n_noise']].values
    ))
    
    fig.update_layout(
        title='<b>Cluster Distribution Over Time</b><br><sub>Number of clusters detected per month in Lyon Flickr data</sub>',
        xaxis_title='Year-Month',
        yaxis_title='Number of Clusters',
        hovermode='x unified',
        height=600,
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    # Save
    fig.write_html(output_file)
    print(f"\n✅ Cluster histogram saved: {output_file}")
    
    return df_stats

def generate_interactive_cluster_html(cleaned_csv, clustering_csv, output_file='../maps/cluster_timeline_interactive.html'):
    """
    on fait un HTML statique avec un histogramme interactif et une carte synchronisée.
    
    Args:
        cleaned_csv: route a csv de données nettoyées
        clustering_csv: route a csv de données de clustering
        output_file: route a csv de resultat html
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not installed. Install with: pip install plotly")
        return
    
    print("Loading data for interactive cluster visualization...")
    
    # charger les données nettoyées
    df_clean = pd.read_csv(cleaned_csv)
    print(f"Loaded {len(df_clean):,} cleaned records")
    
    # charger les resultats du clustering 
    try:
        df_clusters = pd.read_csv(clustering_csv)
        print(f"Loaded {len(df_clusters):,} clustered records")
    except FileNotFoundError:
        print(f"❌ Clustering file not found: {clustering_csv}")
        return
    
    # fusionner les données
    df_merged = df_clean.merge(
        df_clusters[['id', 'cluster']], 
        on='id', 
        how='inner'
    )
    print(f"Merged {len(df_merged):,} records")
    
    # créer la columne pour le 'datetime'
    df_merged['datetime'] = pd.to_datetime(
        pd.DataFrame({
            'year': df_merged['date_taken_year'],
            'month': df_merged['date_taken_month'],
            'day': df_merged['date_taken_day']
        }),
        errors='coerce'
    )
    
    df_merged = df_merged.dropna(subset=['datetime'])
    df_merged['year_month'] = df_merged['datetime'].dt.to_period('M')
    df_merged['year_month_str'] = df_merged['year_month'].astype(str)
    
    # calculer les statistiques pour chaque mois
    monthly_stats = []
    for period, group in df_merged.groupby('year_month'):
        n_clusters = group['cluster'].nunique()
        n_photos = len(group)
        n_noise = (group['cluster'] == -1).sum()
        n_clustered = n_photos - n_noise
        
        monthly_stats.append({
            'date': period.start_time,
            'year_month': str(period),
            'n_clusters': n_clusters,
            'n_photos': n_photos,
            'n_clustered': n_clustered,
            'n_noise': n_noise
        })
    
    df_stats = pd.DataFrame(monthly_stats)
    
    print(f"\n📊 Cluster Timeline Statistics:")
    print(f"   Date range: {df_stats['date'].min().date()} to {df_stats['date'].max().date()}")
    print(f"   Total months: {len(df_stats)}")
    print(f"   Avg clusters/month: {df_stats['n_clusters'].mean():.1f}")
    print(f"   Max clusters: {df_stats['n_clusters'].max()}")
    print(f"   Min clusters: {df_stats['n_clusters'].min()}")
    
    # créer les données de GeoJSON pour chaque mois
    geojson_by_month = {}
    for month_str, group in df_merged.groupby('year_month_str'):
        features = []
        for idx, row in group.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['long'], row['lat']]
                },
                "properties": {
                    "cluster": int(row['cluster']) if pd.notna(row['cluster']) else -1,
                    "date": row['datetime'].strftime('%Y-%m-%d'),
                    "id": str(row['id'])
                }
            })
        geojson_by_month[month_str] = {
            "type": "FeatureCollection",
            "features": features
        }
    
    # faire cluster avec les couleurs
    all_clusters = df_merged['cluster'].dropna().unique()
    colors_list = px.colors.qualitative.Light24
    cluster_colors = {int(c): colors_list[i % len(colors_list)] for i, c in enumerate(sorted(all_clusters))}
    cluster_colors[-1] = '#999999'  # Gray for noise
    
    # Faire le HTML avec JavaScript dedans
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive Cluster Timeline</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                text-align: center;
                color: #333;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #666;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            .selected-info {{
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 20px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 4px;
            }}
            #histogram {{
                width: 100%;
                height: 500px;
                margin-bottom: 30px;
            }}
            .map-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            #map {{
                width: 100%;
                height: 600px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .instructions {{
                background-color: #e3f2fd;
                padding: 15px;
                border-left: 4px solid #2196F3;
                margin-bottom: 20px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive Cluster Timeline</h1>
            <p class="subtitle">Explore clusters over time - click bars to filter the map</p>
            
            <div class="instructions">
                <strong>How to use:</strong> Click on any bar in the histogram to filter the map to show only clusters from that month. 
                The map will update automatically to show photo locations colored by their cluster assignment.
            </div>
            
            <div class="selected-info" id="selected-info">
                Click on a histogram bar to view clusters for a specific month
            </div>
            
            <div id="histogram"></div>
            
            <div class="map-title" id="map-title">All Clusters</div>
            <div id="map"></div>
        </div>
        
        <script>
            // Data
            const statsData = {json.dumps(df_stats.astype({'date': str}).to_dict('records'))};
            const mergedData = {json.dumps(df_merged[['lat', 'long', 'cluster', 'year_month_str', 'datetime']].fillna(-1).astype({'datetime': str}).to_dict('records'))};
            const clusterColors = {json.dumps(cluster_colors)};
            const geojsonByMonth = {json.dumps(geojson_by_month)};
            
            let map = null;
            let selectedMonth = null;
            
            // Initialize map
            function initMap() {{
                if (map) map.remove();
                
                const center = [45.75, 4.85];
                map = L.map('map').setView(center, 13);
                
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 19
                }}).addTo(map);
                
                return map;
            }}
            
            // Update map with data
            function updateMap(month = null) {{
                // Clear existing markers
                map.eachLayer(layer => {{
                    if (layer instanceof L.CircleMarker) {{
                        map.removeLayer(layer);
                    }}
                }});
                
                // Filter data
                let dataToShow = mergedData;
                if (month) {{
                    dataToShow = mergedData.filter(d => d.year_month_str === month);
                }}
                
                // Add markers
                dataToShow.forEach(row => {{
                    const cluster = parseInt(row.cluster);
                    const color = clusterColors[cluster] || '#999999';
                    
                    L.circleMarker([row.lat, row.long], {{
                        radius: 5,
                        fillColor: color,
                        color: color,
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.7,
                        popup: `<b>Cluster:</b> ${{cluster}}<br><b>Date:</b> ${{row.datetime}}`
                    }}).addTo(map);
                }});
                
                // Center map on data
                if (dataToShow.length > 0) {{
                    const lats = dataToShow.map(d => d.lat);
                    const longs = dataToShow.map(d => d.long);
                    const bounds = L.latLngBounds(
                        [Math.min(...lats), Math.min(...longs)],
                        [Math.max(...lats), Math.max(...longs)]
                    );
                    map.fitBounds(bounds, {{ padding: [50, 50] }});
                }}
            }}
            
            // Create histogram
            const trace = {{
                x: statsData.map(s => s.year_month),
                y: statsData.map(s => s.n_clusters),
                type: 'bar',
                marker: {{
                    color: statsData.map(s => s.n_clusters),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{title: 'Clusters'}}
                }},
                hovertemplate: '<b>%{{x}}</b><br>Clusters: %{{y}}<br>Photos: %{{customdata[0]}}<br>Clustered: %{{customdata[1]}}<br>Noise: %{{customdata[2]}}<extra></extra>',
                customdata: statsData.map(s => [s.n_photos, s.n_clustered, s.n_noise])
            }};
            
            const layout = {{
                title: 'Cluster Count Over Time (Hover a bar to filter)',
                xaxis: {{ title: 'Year-Month' }},
                yaxis: {{ title: 'Number of Clusters' }},
                hovermode: 'x unified',
                height: 500
            }};
            
            Plotly.newPlot('histogram', [trace], layout, {{ responsive: true }});
            
            // Handle hovers on histogram
            document.getElementById('histogram').on('plotly_hover', function(data) {{
                const point = data.points[0];
                const pointIndex = point.pointNumber;
                const monthData = statsData[pointIndex];
                
                if (monthData) {{
                    selectedMonth = monthData.year_month;
                    
                    // Update info
                    document.getElementById('selected-info').innerHTML = 
                        `<strong>${{monthData.year_month}}</strong> - ${{monthData.n_clusters}} clusters, ${{monthData.n_photos}} photos`;
                    
                    // Update map title
                    document.getElementById('map-title').innerHTML = `Clusters in ${{monthData.year_month}}`;
                    
                    // Update map
                    updateMap(selectedMonth);
                }}
            }});
            
            // Initialize
            initMap();
            updateMap();
        </script>
    </body>
    </html>
    """
    
    # Save HTML
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n Interactive HTML generated!")
    print(f"   File: {output_file}")
    print(f"   Open in your browser to interact with the timeline and map")
    
    return df_stats

if __name__ == "__main__":
    # Generate both static and temporal maps
    generate_lyon_map('../data/flickr_data2_cleaned.csv')
    print("\n" + "="*70)