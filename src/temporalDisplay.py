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

import calendar
from collections import Counter

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

def analyze_seasonal_trends(csv_path, output_file='../maps/seasonal_trends.html'):
    """
    Analyse les tendances saisonnières du tourisme à Lyon.
    Identifie les mois de haute/basse saison et les patterns récurrents.
    
    Args:
        csv_path: Chemin vers le CSV de données nettoyées
        output_file: Fichier HTML de sortie
    
    Returns:
        DataFrame avec statistiques mensuelles
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not installed. Install with: pip install plotly")
        return
    
    print("\n" + "="*70)
    print(" "*15 + "ANALYSE SAISONNIÈRE - LYON FLICKR")
    print("="*70)
    
    # Charger les données
    df = pd.read_csv(csv_path)
    print(f"\n📊 Chargement de {len(df):,} photos...")
    
    # Créer datetime
    df['datetime'] = pd.to_datetime(
        pd.DataFrame({
            'year': df['date_taken_year'],
            'month': df['date_taken_month'],
            'day': df['date_taken_day']
        }),
        errors='coerce'
    )
    df = df.dropna(subset=['datetime'])
    
    # Extraire mois et année
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['month_name'] = df['datetime'].dt.month_name()
    
    # Statistiques globales par mois (tous les ans agrégés)
    monthly_counts = df.groupby('month').size()
    monthly_avg = monthly_counts.mean()
    monthly_std = monthly_counts.std()
    
    # Identifier les mois de haute saison (> moyenne + 0.5 * std)
    high_season_threshold = monthly_avg + (0.5 * monthly_std)
    low_season_threshold = monthly_avg - (0.5 * monthly_std)
    
    high_season_months = monthly_counts[monthly_counts > high_season_threshold]
    low_season_months = monthly_counts[monthly_counts < low_season_threshold]
    
    # Statistiques par mois et par année
    monthly_yearly = df.groupby(['year', 'month']).size().reset_index(name='count')
    
    print(f"\n📈 Statistiques saisonnières :")
    print(f"   Période analysée : {df['year'].min()} - {df['year'].max()}")
    print(f"   Total de photos : {len(df):,}")
    print(f"   Moyenne mensuelle : {monthly_avg:.0f} photos")
    print(f"   Écart-type : {monthly_std:.0f}")
    
    print(f"\n🌞 HAUTE SAISON (> {high_season_threshold:.0f} photos/mois) :")
    for month_num in high_season_months.index:
        month_name = calendar.month_name[month_num]
        count = high_season_months[month_num]
        print(f"   • {month_name:12s} : {count:,} photos (+{((count/monthly_avg - 1)*100):.1f}%)")
    
    print(f"\n❄️  BASSE SAISON (< {low_season_threshold:.0f} photos/mois) :")
    for month_num in low_season_months.index:
        month_name = calendar.month_name[month_num]
        count = low_season_months[month_num]
        print(f"   • {month_name:12s} : {count:,} photos ({((count/monthly_avg - 1)*100):.1f}%)")
    
    # Créer le graphique principal
    month_names = [calendar.month_name[i] for i in range(1, 13)]
    colors = ['#FF6B6B' if monthly_counts[i] > high_season_threshold 
              else '#4ECDC4' if monthly_counts[i] < low_season_threshold 
              else '#95A5A6' for i in range(1, 13)]
    
    fig = go.Figure()
    
    # Barres mensuelles
    fig.add_trace(go.Bar(
        x=month_names,
        y=monthly_counts.values,
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=monthly_counts.values,
        textposition='outside',
        texttemplate='%{text:,}',
        name='Photos par mois',
        hovertemplate='<b>%{x}</b><br>' +
                      'Photos: %{y:,}<br>' +
                      '<extra></extra>'
    ))
    
    # Ligne de moyenne
    fig.add_hline(
        y=monthly_avg, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"Moyenne ({monthly_avg:.0f})",
        annotation_position="right"
    )
    
    # Zones haute/basse saison
    fig.add_hrect(
        y0=high_season_threshold, 
        y1=monthly_counts.max() * 1.1,
        fillcolor="red", 
        opacity=0.1,
        annotation_text="Haute saison",
        annotation_position="top left"
    )
    
    fig.add_hrect(
        y0=0, 
        y1=low_season_threshold,
        fillcolor="blue", 
        opacity=0.1,
        annotation_text="Basse saison",
        annotation_position="bottom left"
    )
    
    fig.update_layout(
        title='<b>Saisonnalité du Tourisme à Lyon</b><br><sub>Nombre de photos Flickr par mois (toutes années confondues)</sub>',
        xaxis_title='Mois',
        yaxis_title='Nombre de photos',
        height=600,
        template='plotly_white',
        showlegend=False,
        font=dict(size=12)
    )
    
    # Sauvegarder
    fig.write_html(output_file)
    print(f"\n✅ Graphique sauvegardé : {output_file}")
    
    # Créer un second graphique : évolution année par année
    fig2 = go.Figure()
    
    for year in sorted(df['year'].unique()):
        year_data = monthly_yearly[monthly_yearly['year'] == year]
        # Créer un dataframe complet avec tous les mois (remplir les manquants avec 0)
        all_months = pd.DataFrame({'month': range(1, 13)})
        year_data_full = all_months.merge(year_data[['month', 'count']], on='month', how='left').fillna(0)
        
        fig2.add_trace(go.Scatter(
            x=month_names,
            y=year_data_full['count'],
            mode='lines+markers',
            name=str(year),
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig2.update_layout(
        title='<b>Évolution Mensuelle par Année</b><br><sub>Tendances saisonnières comparées année par année</sub>',
        xaxis_title='Mois',
        yaxis_title='Nombre de photos',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            title='Année',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )
    
    output_file2 = output_file.replace('.html', '_yearly.html')
    fig2.write_html(output_file2)
    print(f"✅ Graphique annuel sauvegardé : {output_file2}")
    
    print("\n" + "="*70)
    
    return monthly_counts

def detect_special_events(csv_path, threshold_multiplier=2.5, output_file='../maps/special_events.html'):
    """
    Détecte les pics d'activité anormaux correspondant à des événements spéciaux.
    Identifie les jours avec un nombre de photos significativement supérieur à la moyenne.
    
    Args:
        csv_path: Chemin vers le CSV de données nettoyées
        threshold_multiplier: Multiplicateur du seuil (défaut: 2.5 écarts-types)
        output_file: Fichier HTML de sortie
    
    Returns:
        DataFrame avec les événements détectés
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not installed. Install with: pip install plotly")
        return
    
    print("\n" + "="*70)
    print(" "*15 + "DÉTECTION D'ÉVÉNEMENTS SPÉCIAUX - LYON")
    print("="*70)
    
    # Charger les données
    df = pd.read_csv(csv_path)
    print(f"\n🔍 Analyse de {len(df):,} photos...")
    
    # Créer datetime
    df['datetime'] = pd.to_datetime(
        pd.DataFrame({
            'year': df['date_taken_year'],
            'month': df['date_taken_month'],
            'day': df['date_taken_day']
        }),
        errors='coerce'
    )
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.date
    
    # Compter photos par jour
    daily_counts = df.groupby('date').size()
    
    # Statistiques
    mean_daily = daily_counts.mean()
    std_daily = daily_counts.std()
    median_daily = daily_counts.median()
    threshold = mean_daily + (threshold_multiplier * std_daily)
    
    print(f"\n📊 Statistiques quotidiennes :")
    print(f"   Moyenne : {mean_daily:.1f} photos/jour")
    print(f"   Médiane : {median_daily:.0f} photos/jour")
    print(f"   Écart-type : {std_daily:.1f}")
    print(f"   Seuil de détection : {threshold:.0f} photos/jour ({threshold_multiplier} × σ)")
    
    # Détecter anomalies
    anomalies = daily_counts[daily_counts > threshold].sort_values(ascending=False)
    
    print(f"\n🎉 {len(anomalies)} événements spéciaux détectés :")
    print("="*70)
    
    # Analyser chaque événement
    events = []
    for date, count in anomalies.items():
        day_df = df[df['date'] == date]
        
        # Extraire mots-clés du jour
        if 'text_merged' in day_df.columns:
            text_merged = ' '.join(day_df['text_merged'].dropna().astype(str))
            # Nettoyer et compter les mots
            words = text_merged.lower().split()
            # Filtrer les mots trop courts
            words = [w for w in words if len(w) > 3]
            top_words = Counter(words).most_common(10)
            keywords = ', '.join([f"{w[0]} ({w[1]})" for w in top_words[:5]])
        else:
            keywords = "N/A"
        
        # Statistiques géographiques
        lat_mean = day_df['lat'].mean()
        long_mean = day_df['long'].mean()
        
        events.append({
            'date': date,
            'day_name': pd.to_datetime(date).strftime('%A'),
            'photos': count,
            'ratio': count / mean_daily,
            'keywords': keywords,
            'lat': lat_mean,
            'long': long_mean,
            'n_users': day_df['user'].nunique() if 'user' in day_df.columns else 0
        })
        
        # Afficher les 20 premiers événements
        if len(events) <= 20:
            print(f"\n📅 {date} ({pd.to_datetime(date).strftime('%A')}) :")
            print(f"   • Photos : {count:,} ({count/mean_daily:.1f}× la moyenne)")
            print(f"   • Utilisateurs : {events[-1]['n_users']}")
            print(f"   • Mots-clés : {keywords[:100]}...")
    
    events_df = pd.DataFrame(events)
    
    # Afficher résumé
    if len(events) > 20:
        print(f"\n... et {len(events) - 20} autres événements (voir graphique)")
    
    # Identifier patterns
    print(f"\n📈 Analyse des patterns :")
    
    # Événements par jour de la semaine
    day_counts = events_df['day_name'].value_counts()
    print(f"\n   Jours les plus fréquents :")
    for day, count in day_counts.head(3).items():
        print(f"   • {day:10s} : {count} événements")
    
    # Événements par mois
    events_df['month'] = pd.to_datetime(events_df['date']).dt.month
    month_counts = events_df['month'].value_counts().sort_index()
    print(f"\n   Mois les plus actifs :")
    for month, count in month_counts.head(3).items():
        month_name = calendar.month_name[month]
        print(f"   • {month_name:10s} : {count} événements")
    
    # Créer visualisation
    fig = go.Figure()
    
    # Timeline des événements
    fig.add_trace(go.Scatter(
        x=events_df['date'],
        y=events_df['photos'],
        mode='markers',
        marker=dict(
            size=events_df['photos'] / 20,  # Taille proportionnelle
            color=events_df['ratio'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="× Moyenne"),
            line=dict(width=1, color='darkred')
        ),
        text=events_df['keywords'].str[:50],
        hovertemplate='<b>%{x}</b><br>' +
                      'Photos: %{y:,}<br>' +
                      'Ratio: %{marker.color:.1f}×<br>' +
                      'Mots-clés: %{text}<br>' +
                      '<extra></extra>',
        name='Événements'
    ))
    
    # Ligne de seuil
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Seuil ({threshold:.0f})",
        annotation_position="right"
    )
    
    # Ligne de moyenne
    fig.add_hline(
        y=mean_daily,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Moyenne ({mean_daily:.0f})",
        annotation_position="right"
    )
    
    fig.update_layout(
        title='<b>Détection d\'Événements Spéciaux à Lyon</b><br><sub>Pics d\'activité photographique anormaux</sub>',
        xaxis_title='Date',
        yaxis_title='Nombre de photos',
        height=600,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Sauvegarder
    fig.write_html(output_file)
    print(f"\n✅ Graphique sauvegardé : {output_file}")
    
    # Sauvegarder le tableau des événements
    csv_output = output_file.replace('.html', '.csv')
    events_df.to_csv(csv_output, index=False)
    print(f"✅ Tableau des événements sauvegardé : {csv_output}")
    
    # Top 10 événements
    print(f"\n🏆 TOP 10 ÉVÉNEMENTS (photos) :")
    print("="*70)
    for idx, row in events_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['date']} - {row['photos']:,} photos ({row['ratio']:.1f}×)")
        print(f"    {row['keywords'][:80]}...")
    
    print("\n" + "="*70)
    
    return events_df

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "VISUALISATION TEMPORELLE - LYON FLICKR")
    print("="*70 + "\n")
    
    print("Choisissez une visualisation :")
    print("1. Carte statique simple (MarkerCluster)")
    print("2. Carte temporelle avec couches par mois")
    print("3. Histogramme des clusters par mois (Plotly)")
    print("4. Visualisation interactive : histogramme + carte synchronisée")
    print("5. Analyse saisonnière (tendances mensuelles)")
    print("6. Détection d'événements spéciaux (pics d'activité)")
    print("7. Tout générer")
    
    choice = input("\nVotre choix (1-7) : ").strip()
    
    csv_path = '../data/flickr_data2_cleaned.csv'
    
    if choice == '1':
        generate_lyon_map(csv_path)
    
    elif choice == '2':
        sample = input("Échantillonner les données ? (appuyez sur Entrée pour non, ou entrez la taille) : ").strip()
        sample_size = int(sample) if sample else None
        generate_temporal_lyon_map(csv_path, time_freq='MS', sample_size=sample_size)
    
    elif choice == '3':
        if not PLOTLY_AVAILABLE:
            print("\n❌ Plotly n'est pas installé. Installez avec : pip install plotly")
        else:
            clustering_file = input("Fichier de clustering (appuyez sur Entrée pour '../data/flickr_data2_hierarchical_complete.csv') : ").strip()
            if not clustering_file:
                clustering_file = '../data/flickr_data2_hierarchical_complete.csv'
            generate_cluster_histogram(csv_path, clustering_file)
    
    elif choice == '4':
        if not PLOTLY_AVAILABLE:
            print("\n❌ Plotly n'est pas installé. Installez avec : pip install plotly")
        else:
            clustering_file = input("Fichier de clustering (appuyez sur Entrée pour '../data/flickr_data2_hierarchical_complete.csv') : ").strip()
            if not clustering_file:
                clustering_file = '../data/flickr_data2_hierarchical_complete.csv'
            generate_interactive_cluster_html(csv_path, clustering_file)
    
    elif choice == '5':
        if not PLOTLY_AVAILABLE:
            print("\n❌ Plotly n'est pas installé. Installez avec : pip install plotly")
        else:
            analyze_seasonal_trends(csv_path)
    
    elif choice == '6':
        if not PLOTLY_AVAILABLE:
            print("\n❌ Plotly n'est pas installé. Installez avec : pip install plotly")
        else:
            threshold = input("Seuil de détection (appuyez sur Entrée pour 2.5) : ").strip()
            threshold_multiplier = float(threshold) if threshold else 2.5
            detect_special_events(csv_path, threshold_multiplier=threshold_multiplier)
    
    elif choice == '7':
        print("\n🚀 Génération de toutes les visualisations...\n")
        
        print("[1/6] Carte statique...")
        generate_lyon_map(csv_path)
        
        print("\n[2/6] Carte temporelle (échantillon de 5000 points)...")
        generate_temporal_lyon_map(csv_path, time_freq='MS', sample_size=5000)
        
        if PLOTLY_AVAILABLE:
            clustering_file = '../data/flickr_data2_hierarchical_complete.csv'
            
            print("\n[3/6] Histogramme des clusters...")
            generate_cluster_histogram(csv_path, clustering_file)
            
            print("\n[4/6] Visualisation interactive...")
            generate_interactive_cluster_html(csv_path, clustering_file)
            
            print("\n[5/6] Analyse saisonnière...")
            analyze_seasonal_trends(csv_path)
            
            print("\n[6/6] Détection d'événements...")
            detect_special_events(csv_path)
        else:
            print("\n⚠ Plotly non disponible, visualisations 3-6 ignorées")
        
        print("\n✅ Toutes les visualisations ont été générées !")
    
    else:
        print("❌ Choix invalide")
    
    print("\n" + "="*70)