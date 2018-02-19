# -*- coding: utf-8 -*-
from json import dumps
import shapefile
import click

@click.command()
@click.option(
    '--f',
    help="Filename",
    default=r'Data\GeoData\vg2500_bld.shp'
)
def main(f):
    # read the shapefile
    reader = shapefile.Reader(f)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []
    for sr in reader.shapeRecords():
        record = sr.record
        # Make sure everything is utf-8 compatable
        record = [r.decode('utf-8', 'ignore') if isinstance(r, bytes)
                  else r for r in record]
        atr = dict(zip(field_names, record))
        geom = sr.shape.__geo_interface__
        buffer.append(dict(type="Feature", geometry=geom, properties=atr))

    # write the GeoJSON file
    with open('_'.join(f.split('.')[:-1]) + ".geo.json", "w") as geojson:
        geojson.write(dumps({"type": "FeatureCollection",\
                             "features": buffer}, indent=2) + "\n")

if __name__ == '__main__':
    main()