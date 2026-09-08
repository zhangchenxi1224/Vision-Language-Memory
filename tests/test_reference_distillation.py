import unittest
import torch
from vision_memory.training.empirical_bank_flow import EmpiricalBankFlow
from vision_memory.training.reference_distillation import reference_trajectory,rollout_distillation_loss


class ReferenceDistillationTest(unittest.TestCase):
    def test_reference_four_steps_and_stopped_gradients(self):
        source=torch.zeros(1,2,2,requires_grad=True)
        noise=torch.ones_like(source,requires_grad=True)
        target=torch.full((1,1,2,2),.2,requires_grad=True)
        ref=reference_trajectory(EmpiricalBankFlow(source,target),source,noise)
        self.assertEqual(len(ref),5)
        torch.testing.assert_close(ref[0],.5*noise)
        torch.testing.assert_close(ref[-1],target[0])
        self.assertFalse(any(x.requires_grad for x in ref))

    def test_loss_reaches_every_student_step_not_teacher(self):
        p=torch.tensor(.3,requires_grad=True)
        target=torch.tensor(.1,requires_grad=True)
        states=[p*0]+[p*i for i in range(1,5)]
        refs=[target for _ in range(5)]
        loss,end,path=rollout_distillation_loss(states,refs)
        torch.testing.assert_close(loss,end+path)
        loss.backward()
        self.assertGreater(abs(float(p.grad)),0)
        self.assertIsNone(target.grad)

    def test_exact_path_has_zero_loss(self):
        states=[torch.zeros(1,2) for _ in range(5)]
        self.assertEqual(float(rollout_distillation_loss(states,states)[0]),0.)


if __name__=='__main__': unittest.main()
