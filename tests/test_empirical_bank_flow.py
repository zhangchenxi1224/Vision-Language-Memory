import unittest
import torch
from vision_memory.training.empirical_bank_flow import EmpiricalBankFlow


class EmpiricalBankFlowTest(unittest.TestCase):
    def setUp(self):
        g=torch.Generator().manual_seed(710)
        self.source=torch.randn((1,2,3),generator=g,dtype=torch.float64)
        self.teachers=torch.randn((5,1,2,3),generator=g,dtype=torch.float64)
        self.state=torch.randn((1,2,3),generator=g,dtype=torch.float64)

    def test_posterior_agrees_with_direct_gaussian_densities(self):
        f=EmpiricalBankFlow(self.source,self.teachers)
        sigma=.37
        means=sigma*self.source+(1-2*sigma)*self.teachers
        logp=-((self.state-means)/sigma).flatten(1).square().sum(1)/2
        v,w,var=f.evaluate(self.state,sigma)
        torch.testing.assert_close(w,logp.softmax(0))
        conditional_v=(self.state-self.teachers)/sigma
        torch.testing.assert_close(v,(w[:,None,None,None]*conditional_v).sum(0))
        torch.testing.assert_close(var,(w*(conditional_v-v).flatten(1).square().mean(1)).sum())

    def test_start_variance_decomposition_and_uniform_posterior(self):
        f=EmpiricalBankFlow(self.source,self.teachers)
        v,w,var=f.evaluate(self.state,.5)
        torch.testing.assert_close(w,torch.full((5,),.2,dtype=torch.float64))
        prediction=torch.zeros_like(v)
        raw_mse=((self.state-self.teachers)/.5-prediction).square().mean()
        torch.testing.assert_close(raw_mse,var+(v-prediction).square().mean())

    def test_single_target_four_steps_reaches_target(self):
        f=EmpiricalBankFlow(self.source,self.teachers[:1])
        x=self.state.clone()
        for sigma in [.5,.375,.25,.125]:
            v,_,var=f.evaluate(x,sigma)
            x=x-.125*v
            self.assertAlmostEqual(float(var),0.)
        torch.testing.assert_close(x,self.teachers[0])

    def test_target_order_does_not_change_vector_field(self):
        a=EmpiricalBankFlow(self.source,self.teachers)
        b=EmpiricalBankFlow(self.source,self.teachers.flip(0))
        torch.testing.assert_close(a.evaluate(self.state,.2)[0],b.evaluate(self.state,.2)[0])


if __name__=='__main__':
    unittest.main()
