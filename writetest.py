import numpy as np
from datetime import datetime
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
    info2.data_source.sample.comment='this is a comment'

    ds=fileio.OrsoDataset(info, data)
    ds2=fileio.OrsoDataset(info2, data)

    info3=fileio.Orso(
        creator=fileio.Creator(
            name='Artur Glavic',
            affiliation='Paul Scherrer Institut',
            time=datetime.now(),
            computer='localhost'),
        data_source=fileio.DataSource(
            sample=fileio.Sample(
                name='My Sample',
                type='solid',
                description='Something descriptive',
            ),
            experiment=fileio.Experiment(
                title='Main experiment',
                instrument='Reflectometer',
                date=datetime.now(),
                probe='x-rays',
            ),
            owner=fileio.Person('someone', 'important'),
            measurement=fileio.Measurement(
                instrument_settings=fileio.InstrumentSettings(
                    incident_angle=fileio.Value(13.4, 'deg'),
                    wavelength=fileio.Value(5.34, 'A')
                ),
                data_files=['abc', 'def', 'ghi'],
                references=['more', 'files'],
                scheme='angle-dispersive',
            ),
        ),
        reduction=fileio.Reduction(software='awesome orso'),
        data_set='Filled header',
        columns=info.columns,
    )
    ds3=fileio.OrsoDataset(info3, data)

    fileio.save_orso([ds, ds2, ds3], 'test.ort')

    ls1, ls2, ls3=fileio.load_orso('test.ort')
    print(ls1==ds, ls2==ds2, ls3==ds3)

if __name__=='__main__':
    main()