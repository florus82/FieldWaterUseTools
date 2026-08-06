INT_TO_MONTH = {
    '01': 'January',
    '02': 'February',
    '03': 'March',
    '04': 'April',
    '05': 'May',
    '06': 'June',
    '07': 'July',
    '08': 'August',
    '09': 'September',
    '10': 'October',
    '11': 'November',
    '12': 'December'
    }

REAL_INT_TO_MONTH = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
    }

MONTH_TO_02D = {v: k for k, v in INT_TO_MONTH.items()}

DAYCOUNT_LEAP = [31,29,31,30,31,30,31,31,30,31,30,31]
DAYCOUNT_NOLEAP = [31,28,31,30,31,30,31,31,30,31,30,31]

DRY_ADIABAT = 0.0098 # K/m
STANDARD_ADIABAT = 0.0065 # K/m
MOIST_ADIABAT = 0.002 # K/m

GRAVITY = 9.80665 # m/s**2
MOLAR_MASS_AIR = 0.0289644 # kg/mol
UNIVERSAL_GAS = 8.3144598 # J/(mol*K)

SOLAR_CONST = 1367.0 # Solar constant (W/m^2)


# for thuenen masking
VALID_AGRO_VALUES = [
    200, # Permanent grassland
    1101,# Winter wheat
    1102, # Winter barley
    1103, # Winter rye
    1201, # Spring barley
    1202, # Spring oat
    1300, # Maize
    1401, # Potato
    1402, # Sugar beet
    1501, # Winter rapeseed
    1502, # Sunflower
    1602, # Cultivated grassland
    1603, # Vegetables
    1611, # Peas
    1612, # Broad bean
    1613, # Lupin
    1614, # Soy
    #3001, # Small woody features
    3002, # Other agricultural areas
    3003, # Fallow land
    #3004, # Other areas
    #3011, # Small woody features on other land
    #4001, # Grapevine
    4002, # Hops
    4003, # Orchard
]


# for IACS masking
EXCLUDE_LIST = ['',
 '(Beta-)Rübensamenvermehrung',
 'Alle anderen Flächen (keine LF)',
 'Baumschulen, nicht für Beerenobst',
 'Bestockte Rebfläche',
 'Erosionsschutzstreifen',
 'Forstflächen (Waldbodenflächen)',
 'Gewässerschutzstreifen',
 'Grassamenvermehrung',
 'Haus- und Nutzgärten',
 'KUP lt. Direktzahlungendurchführungsverordnung',
 'KUP lt. Direktzahlungendurchführungsverordnung (keine ÖVF)',
 'KUP lt. GAPDZV',
 'N. LNF, n. Art. 32(2b(i)) der VO(EG) Nr.1307/2013 beihilfef. Fl.',
 'Nicht landwirt. Fl. In der Verfügungsgewalt des Antragstellers, die gem. § 15 (1) DirektZahlDurchfG als umweltsensibles Dauergrünland bestimmt worden sind',
 'Nicht landwirt. Fl. infolge Genehmigung DGL Umwandlung',
 'Nicht landwirtschaftliche, aber nach Art. 32(2b (i)) der VO (EG) Nr. 1307/2013 beihilfefähige Fläche',
 'Nicht landwirtschaftliche, aber nach Art. 32(2b (i)) der VO (EG) Nr. 1307/2013 beihilfefähige Fläche (Naturschutzflächen, die 2008 noch beihilfefähig waren)',
 'Nicht landwirtschaftliche, aber nach §11 (1) Nr.3 Bst. a) bb) der GAPDZV förderfähige Fläche (Infolge Anwendung der Wasserrahmenrichtlinie)',
 'Nicht landwirtschaftliche, aber §11 (1) Nr.3 Bst. c) der GAPDZV förderfähige Fläche (Aufforstungsverpflichtung nach VO 1257/1999 oder VO (EG) Nr. 1698/2005 oder VO 1305/2013 oder VO 2021/2115 oder bei Eingehung damit in Einklang stehender öffentlich',
 'None',
 'Pufferstreifen ÖVF DGL',
 'Schonstreifen',
 'Streifen am Waldrand (ohne Produktion) ÖVF',
 'Ufervegetation ÖVF',
 'Unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf AL',
 'Unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf DGL',
 'Unbestockte Rebfläche',
 'Vorübergehende, unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf AL',
 'Vorübergehende, unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf DGL',
 'Weihnachtsbäume',
 'Wildäsungsfläche',
 'afforestation_reforestation',
 'alle anderen Flächen (keine LF)',
 'aufgeforstete Dauergrünlandflächen, weder nach 1257/99 oder VO (EG) Nr. 1698/2005  1305/2013oder VO (EU) Nr.1305/2013',
 'aufgeforstete Flächen (VO1257/1999, 1698/2005, 1305/2013)',
 'greenhouse_foil_film',
 'nach VO 1257/1999 oder VO (EG) Nr. 1698/2005 oder VO 1305/2013 aufgeforstete Flächen',
 'not_known_and_other',
 'nurseries_nursery',
 'tree_wood_forest',
 'unmaintained',
 'vorübergehend unbefestigte Mieten-, Stroh-, Futter oder Dunglagerplätze auf DGL',
 'vorübergehende, unbefestigte Mieten-, Stroh-, Futter oder Dunglagerplätze auf AL',
 'Hecke/Knick (CC-LE)',
 'Hecke/Knick (LE)', 
 'Feldgehölz (CC-LE)', 
 'Feldgehölz (LE)', 
 'Feuchtgebiet (CC-LE)', 
 'Feldrain (LE)', 
 'Feldrain (CC-LE)', 
 'Fels-, Steinriegel, naturversteinte Fläche (CC-LE)', 
 'Trocken-, Natursteinmauer, Lesesteinwall (CC-LE)', 
 'Tümpel, Söll, Doline (CC-LE)',
 'Baumreihe (CC-LE)',
 'Graben (LE)',
 'nicht landw. Fläche']


EXCLUDE_LIST_FTW = ['afforestation_reforestation',
                    'apples',
                    'cherry_cherries',
                    'fallow_land_not_crop',
                    'greenhouse_foil_film',
                    'hazelnuts_hazel',
                    'not_known_and_other',
                    'nurseries_nursery',
                    'orchards_fruits',
                    'other_permanent_crops_plantations',
                    'pasture_meadow_grassland_grass',
                    'tree_wood_forest',
                    'unmaintained',
                    'vineyards_wine_vine_rebland_grapes',
                    'walnuts']