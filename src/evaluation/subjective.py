from scipy import stats
from src.evaluation import eval

# Objective
exploited_sdr = [19.597110164291337, 12.498122028745831, 15.187255038925692, 16.73324828941119, 14.890488828281676, 17.95433691990165, 10.433497686008582, 16.736643232670502, 15.69696662576665, 14.48198525709895, 12.992140159230344, 16.339738135607014]
exploited_l1 = [1.7650424020623532, 1.6841228307000373, 1.464490357144854, 1.2606325971485592, 1.1423846768108736, 1.559283067303536, 2.372350676207962, 1.149631333810908, 1.7447771591926458, 1.743389224815681, 2.363747364231907, 1.5861185942100464]
unexploited_sdr = [19.24214394090644, 12.181851490920986, 14.604145815839347, 16.490120788964646, 14.5541529963989, 17.566290222716898, 10.743537224454297, 16.713220852980132, 14.270390037183425, 13.348452819421233, 12.65696819911965, 16.337209237371518]
unexploited_l1 = [1.2013523278155351, 1.3908311171505876, 1.3049176781379108, 1.1542213073918812, 0.9204565849692665, 1.4755047927386218, 2.48278056697362, 1.1825043472492776, 1.6714671726423735, 1.7681853898715665, 2.05113026064893, 1.3218159293689176]
musdb_sdr = [13.425036083143482, 7.520047468602935, 11.236222510597123, 13.077350560373635, 9.039813654847553, 11.681837750047915, 8.46538680631524, 14.002411042651318, 10.702908708594027, 10.231138298805647, 9.960663932901289, 12.456845537037442]
musdb_l1 = [1.7939888631303984, 5.118900458589026, 2.9458403672724494, 1.7896408899590628, 2.8466087033483656, 3.6087891780622483, 3.336152483369485, 2.695378224555682, 3.21901351855133, 3.5078522346540666, 3.5285854156480867, 4.240330798732824]

gate30_sdr = [7.798528504048124, 8.053491141481109, 7.050367362945591, 9.159865727699682, 10.075623514901903, 11.850701887229826, 7.042955565204473, 12.579374894765724, 6.798768587902154, 10.886007159993053, 10.975670481892587, 11.433997545434329]
gate30_l1 = [4.496011552848558, 2.503409865970113, 1.8729858877988872, 1.3112980830792085, 1.7562564655432116, 1.2373496217824933, 1.6829880944265445, 0.7974166381537581, 1.776017699343132, 1.1046538929213439, 1.5173013937703275, 1.86283019398782]
gate40_sdr = [7.764224085860775, 7.934463492322264, 6.958656931220521, 9.004486572341676, 9.871171338439012, 11.825925665615253, 6.974191855530176, 12.507467109221382, 6.6584775694837175, 10.881542654639674, 10.944988462894825, 11.316756090130255]
gate40_l1 = [4.556965874785643, 2.7075521286693256, 1.9603124805798549, 1.3963465423801733, 1.9507597868760054, 1.195040102170373, 1.6707378023473825, 0.8119580303219763, 1.900920610440613, 1.0353860932792218, 1.5742935722984193, 1.993593436537873]

exploited_gate40_sdr = [19.756150239176534, 12.498134220234524, 15.198878216127735, 16.729081186944278, 14.896820087075216, 17.956366753163312, 10.433905847800993, 16.744519879201995, 15.700087987916323, 14.481358057336582, 12.99367154464852, 16.335098953750762]
exploited_gate40_l1 = [0.5001405308460573, 1.1707369021314373, 1.2098552725072165, 1.2492980137048313, 0.8614704846426672, 1.5144894748532058, 2.483172863524121, 1.0774384180103638, 1.6366860138355355, 1.7219190184767552, 2.3091322956453686, 1.4984438710848762]

# Corrupting
noise20_sdr = [19.25815903109321, 12.340514439790871, 14.760779799674058, 16.423064613550586, 14.767820172048143, 17.725533867285264, 10.337629709947212, 16.55008532983357, 15.467921516944552, 14.385719659643367, 12.915839767204634, 16.212341899774767]
noise20_l1 = [5.396156922249037, 7.8843765415900995, 8.378992167555763, 5.13990116988714, 6.177770252396245, 6.2567092101032395, 11.98223910279114, 6.054098682185334, 7.793481722953618, 6.650463367838229, 8.189556466842125, 4.260604644737409]
noise40_sdr = [18.341929175245898, 11.98488353577426, 13.939386838373238, 15.699529088533877, 14.455211186987778, 17.180858831580082, 10.093870551146862, 15.999186745743376, 14.794748016734038, 14.156516947791623, 12.739306006931443, 15.813303175318495]
noise40_l1 = [9.29844469005277, 11.119257257188213, 11.828565362760305, 8.501299947858561, 9.813334170464701, 9.03956036979343, 15.9195271712792, 9.900643077846652, 11.552009004716746, 9.449433313642286, 10.954501257459626, 7.345305817204548]
shift25_sdr = [15.973231745476117, 12.135929509015241, 13.775066085712911, 14.336736210371523, 14.666370462744245, 17.352441335387844, 10.191624185607935, 15.967532215125683, 13.648464385853858, 11.958741631207612, 12.542402655288015, 16.252769889094832]
shift25_l1 = [2.141055756342576, 1.820100339774799, 1.6001469270770674, 1.3786826152915168, 1.216021351498203, 1.6377073784652083, 2.379493997391809, 1.1604483793849805, 1.8760524408194104, 1.8236552431692856, 2.3988123608670087, 1.6338314017800821]
shift50_sdr = [16.159944612550113, 12.060562301211961, 13.485346185264998, 14.248154280369906, 14.610240813345763, 17.103980262896265, 10.15652791125927, 15.95760122549458, 14.223168436300844, 11.584096484070024, 12.347240847662666, 16.191395993240793]
shift50_l1 = [2.1994526378878763, 1.7903102786873204, 1.612310278742145, 1.397710578296639, 1.2270721680054852, 1.625448831444851, 2.408733951070655, 1.1877124516105926, 1.847144006937648, 1.8265981238848286, 2.424099366954341, 1.6561584195522288]
zero_right_sdr = [9.251572121581795, 9.265106878401483, 8.801140966591698, 11.360219593151118, 11.882466115789203, 14.440857903273098, 8.027156938482092, 13.531812527456006, 9.137892862587016, 11.001534106037834, 11.21230286712111, 13.163131164055773]
zero_right_l1 = [4.216858511974691, 2.324634457715575, 2.0675572612227504, 1.4201317327791192, 1.732026157010942, 1.6216053206163508, 2.493366918083281, 1.139657883764427, 1.9251131878268524, 1.643651369075653, 2.053894335127023, 1.8996233005690866]

# Subjective
gate_quality = [25,90,90,50,90,25,50,60,65]
gate_bleed = [25,20,30,20,10,15,20,15, 40,10,15,5]
exploited_quality = [85, 50, 80, 60, 85, 40, 75, 80, 60]
exploited_bleed = [60,40,50,40,40,40,50,25, 60, 65, 50, 65]
unexploited_quality = [20,10,80,50,75,90, 30, 90, 70]
unexploited_bleed = [60,80,80,50,50,60,75,95, 50, 70, 80, 85]
musdb_quality = [20,20,70,55,60,80, 35, 45, 50]
musdb_bleed = [50,60,45,70,50,40,90,70, 40, 50, 50, 45]
moises_quality = [100,100,100,100,100,100, 95, 95, 100]
moises_bleed = [90,80,100,100,100,95,100,95, 95, 100, 95, 100]
exploited_gate_quality = [70,60,80,60,80,75,60, 65, 65]
exploited_gate_bleed = [70,50,60,55,55,30,70,85, 60, 50, 60, 70]
hidden_reference_quality = [100, 100, 100, 100, 100, 100, 95, 95, 95]
hidden_reference_bleed = [0, 0, 0, 0, 0, 0, 0, 0, 10, 5, 10, 5]

# Gates
gate_30_400_quality = [15,45,60, 15, 20, 80, 30, 45, 40]
gate_30_400_bleed = [30,60,70,100, 10, 30, 60, 100, 60, 65, 65, 100]
gate_40_400_quality = [20,70,80, 50, 60, 75, 50, 40, 40]
gate_40_400_bleed = [30, 65, 50, 100, 70, 60, 50, 95, 45, 70, 60, 75]
exploited_gate_40_400_quality = [25, 65, 80, 35, 80, 75, 50, 60, 85]
exploited_gate_40_400_bleed = [50, 75, 55, 100, 80, 70, 80, 100, 75, 60, 75, 100]




data_arrays = [gate_30_400_bleed, gate_40_400_bleed, musdb_bleed, unexploited_bleed, exploited_bleed,
               exploited_gate_bleed, moises_bleed, hidden_reference_bleed]
method_names = ['gate_30', 'gate_40', 'musdb', 'unexploited', 'exploited', "exploited_plus_gate", "moises",
                "hidden ref."]

e = eval.Eval()
e.plot_p_value_matrix(data_arrays, method_names)

print(stats.ttest_rel(moises_quality, unexploited_quality))

#print(np.average(hidden_reference_bleed))
#print(np.std(hidden_reference_bleed))
#print(np.std(zero_right_sdr))
#print(np.std(zero_right_l1))
