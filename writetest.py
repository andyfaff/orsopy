import numpy as np
from orsopy import fileio

def main():
    info=fileio.Orso.empty()
    info2=fileio.Orso.empty()
    data=np.array([np.arange(100), np.arange(100), np.arange(100)])
    info.columns=[
        fileio.Column('q', '1/A'),
        fileio.Column('R', '1'),
        fileio.Column('sR', '1'),
    ]
    info2.columns=info.columns
    info.data_source.measurement.instrument_settings.polarization='+'
    info2.data_source.measurement.instrument_settings.polarization='-'
    info.data_set='up polarization'
    info2.data_set='down polarization'
    ds=fileio.OrsoDataset(info, data)
    ds2=fileio.OrsoDataset(info2, data)

    fileio.save([ds, ds2], 'test.ort')

    print(fileio.load('test.ort'))

if __name__=='__main__':
    main()