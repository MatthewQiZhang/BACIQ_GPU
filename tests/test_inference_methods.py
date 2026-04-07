from baciq import inference_methods
import pytest
import pandas as pd
import numpy as np
import torch
from numpy.testing import assert_array_equal as aae
from numpy.testing import assert_allclose as aac


def test_get_proteins_and_indices():
    proteins, idx = inference_methods.get_proteins_and_indices(
        pd.DataFrame(
            {'Protein ID': 'a b c d d a a'.split()}
        ))
    aae(proteins, 'a b c d'.split())
    aae(idx, [0, 1, 2, 3, 3, 0, 0])

    proteins, idx = inference_methods.get_proteins_and_indices(
        pd.DataFrame(
            {'Protein ID': 'd c b a d a a'.split()}
        ))
    aae(proteins, 'd c b a'.split())
    aae(idx, [0, 1, 2, 3, 0, 3, 3])

    with pytest.raises(KeyError) as e:
        inference_methods.get_proteins_and_indices(pd.DataFrame())
    assert 'Expected "Protein ID" in input data' in str(e.value)

    proteins, idx = inference_methods.get_proteins_and_indices(
        pd.DataFrame(
            {'Protein ID': []}
        ))
    aae(proteins, [])
    aae(idx, [])


@pytest.fixture
def pymc_model():
    return inference_methods.PYMC_Model(100, 2, 50, 'ch7')


def test_PYMC_Model_init(pymc_model):
    assert pymc_model.samples == 100
    assert pymc_model.chains == 2
    assert pymc_model.tuning == 50
    assert pymc_model.channel == 'ch7'


def test_PYMC_Model_mcmc_sample(pymc_model, mocker):
    # Mock MCMC instance: get_samples returns mu tensor (samples, n_proteins)
    mock_mcmc_instance = mocker.MagicMock()
    mock_mcmc_instance.get_samples.return_value = {
        'mu': torch.zeros(100, 4)
    }
    mock_mcmc_cls = mocker.patch('baciq.inference_methods.MCMC',
                                  return_value=mock_mcmc_instance)
    mock_nuts_cls = mocker.patch('baciq.inference_methods.NUTS')

    proteins, result = pymc_model.mcmc_sample(
        pd.DataFrame({
            'Protein ID': 'a b c d a'.split(),
            'ch7': [1, 2, 3, 4, 5],
            'sum': [10, 11, 12, 13, 14]
        }),
        bin_width=0.2)

    aae(proteins, 'a b c d'.split())

    # Chains run sequentially, so MCMC/NUTS called once per chain
    assert mock_mcmc_cls.call_count == pymc_model.chains
    assert mock_nuts_cls.call_count == pymc_model.chains

    # Check MCMC was constructed with correct parameters each time
    for call in mock_mcmc_cls.call_args_list:
        assert call[1]['num_samples'] == 100
        assert call[1]['warmup_steps'] == 50
        assert call[1]['num_chains'] == 1

    # run() was called once per chain
    assert mock_mcmc_instance.run.call_count == pymc_model.chains

    assert (mock_mcmc_instance.run.call_args[0][1] ==
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)).all()
    assert (mock_mcmc_instance.run.call_args[0][0] ==
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.float32)).all()


def test_PYMC_Model_unmocked_mcmc_sample():
    pymc_model = inference_methods.PYMC_Model(1000, 2, 500, 'ch0')
    proteins, result = pymc_model.mcmc_sample(
        pd.DataFrame({
            'Protein ID': 'b d c a'.split(),
            'ch0': [10, 250, 500, 1000],
            'sum': [1000, 1000, 1000, 1000]}),
        bin_width=0.1)

    aae(proteins, 'b d c a'.split())
    aae(result.sum(axis=1), [2000, 2000, 2000, 2000])  # 1000 samples * 2 chains
    bins = np.array(range(10))/10
    aac(np.sum(result * bins[None, :] / 2000, axis=1),
        [0.38, 0.43, 0.45, 0.66], atol=0.05)

    proteins, result = pymc_model.mcmc_sample(
        pd.DataFrame({
            'Protein ID': ['b', 'd', 'c', 'a']*10,
            'ch0': [10, 250, 500, 750]*10,
            'sum': [1000, 1000, 1000, 1000]*10}),
        bin_width=0.1)

    aae(proteins, 'b d c a'.split())
    aae(result.sum(axis=1), [2000, 2000, 2000, 2000])  # 1000 samples * 2 chains
    aac(np.sum(result * bins[None, :] / 2000, axis=1),
        [0, 0.2, 0.45, 0.7], atol=0.05)


def test_PYMC_Model_fit_histogram(pymc_model, mocker):
    mock_sample = mocker.patch.object(
        inference_methods.PYMC_Model, 'mcmc_sample',
        return_value=('a b c'.split(),
                      np.array([
                          [0, 10, 20, 10, 0],
                          [40, 0, 0, 0, 0],
                          [0, 0, 0, 30, 10]
                      ])))
    result = pymc_model.fit_histogram(pd.DataFrame({
        'Protein ID': 'a b c'.split()}),
                                bin_width=0.2)
    mock_sample.assert_called_with(mocker.ANY, 0.2)

    assert (result.index == 'a b c'.split()).all()
    assert result.index.name == 'Protein ID'
    aac(result.columns.values, [0, 0.2, 0.4, 0.6, 0.8])
    assert (result.values ==
            [
                [0, 10, 20, 10, 0],
                [40, 0, 0, 0, 0],
                [0, 0, 0, 30, 10]
            ]).all()


def test_PYMC_Model_fit_quantiles(pymc_model, mocker):
    return_quants = np.zeros((3, 10000))
    return_quants[0, 1000:1101] = 1  # 100 1's
    return_quants[1, 0] = 1  # first index
    return_quants[2, 9999] = 1  # last index
    mock_sample = mocker.patch.object(
        inference_methods.PYMC_Model, 'mcmc_sample',
        return_value=('a c b'.split(), return_quants))
    result = pymc_model.fit_quantiles(pd.DataFrame({
        'Protein ID': 'a b c'.split()}),
                                [0.025, 0.5, 0.975])
    assert mock_sample.call_args[1]['bin_width'] == 0.0001
    assert (result.index == 'a c b'.split()).all()
    assert result.index.name == 'Protein ID'
    assert (result.columns.values == ['0.025', '0.5', '0.975']).all()
    aac(result.values,
        [
            [0.1002, 0.1050, 0.1098],
            [0, 0, 0],
            [0.9999, 0.9999, 0.9999],
        ])
