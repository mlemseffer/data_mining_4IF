"""
Test automatisé pour main.py avec option 6 (échantillon large de 20k)
"""
import sys
from io import StringIO

# Simuler l'entrée utilisateur : Option 6 (large sample), 15 clusters, confirmation "oui"
sys.stdin = StringIO("6\n15\noui\n")

# Importer et lancer main
from main import main

if __name__ == "__main__":
    main()
