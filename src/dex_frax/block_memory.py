"""Opt-in safe-boundary leases for fixed-size initial IK blocks.

These are forecast credits, not a custom malloc. An opaque first execution
keeps the complete stage estimate. Later executions use measured residency,
fixed-shape workspace and length-dependent result-buffer reservations.
"""
import os
from . import resource_events as events
from .resources import MIB, process_memory


def enabled():
    return os.environ.get('DEX_FRAX_BLOCK_MEMORY') == '1' and events._gate is not None


def workspace_bytes(compiled):
    # CPU executables can report bounded XLA buffers. Native callback growth is
    # additionally covered by the resident-growth envelope and a 128 MiB floor.
    try:
        info = compiled.memory_analysis()
        names=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
               'host_argument_size_in_bytes','host_output_size_in_bytes','host_temp_size_in_bytes')
        return sum(max(0,int(getattr(info,name,0) or 0)) for name in names)
    except (AttributeError,TypeError,ValueError,NotImplementedError,RuntimeError):
        return 0


class InitialBlocks:
    def __init__(self, frames, pose_columns, block_frames, compiled):
        self.stage=events._stage
        self.frames=int(frames);self.columns=int(pose_columns)
        self.block_frames=int(block_frames);self.workspace=workspace_bytes(compiled)
        self.previous_pss=None;self.completed=0;self.growth=0;self.blocks_seen=0

    def result_reserve(self, completed):
        # q (pose_columns) + three record values, all <= FP64.
        # Retained block arrays have already been counted in PSS. Budget the
        # remaining block results, concatenated outputs and self.initial copy.
        row=(self.columns+3)*8
        remaining=max(0,self.frames-int(completed))*row
        concatenation=self.frames*row
        # Concatenation, speed/difference temporaries, initial copy and Python
        # record/flag lists. These buffers are length-dependent even at the tail.
        finalize_extra=self.frames*max(self.columns*8*4,self.columns*8+3*32+128)
        containers=((self.frames+self.block_frames-1)//self.block_frames)*2048
        return remaining+concatenation+finalize_extra+containers

    def checkpoint(self, completed=0, first=False):
        resident=process_memory(os.getpid())
        if self.previous_pss is not None:
            output_growth=max(0,int(completed)-self.completed)*(self.columns+3)*8
            self.blocks_seen+=1
            # First-use initialization is already resident after the first block;
            # it is not a recurring allocation for every subsequent block.
            if self.blocks_seen>1:
                self.growth=max(self.growth,max(0,resident-self.previous_pss-output_growth))
        self.previous_pss=resident;self.completed=int(completed)
        # A positive resident-growth envelope is a measured estimate, not proof
        # of a native allocation bound. First blocks retain conservative grants.
        workspace=max(128*MIB,self.workspace,int(self.growth*1.25)+32*MIB)
        buffers=self.result_reserve(completed)
        final=int(completed)>=self.frames and not first
        events.phase('ik_finalize' if final else 'ik_block',
                     measured_pss_bytes=resident,compile_possible=False,
                     memory_scope='initial_finalize' if final else 'initial_block',
                     first_block=bool(first),frames_total=self.frames,
                     frames_completed=int(completed),next_block_frames=min(self.block_frames,max(0,self.frames-int(completed))),
                     workspace_reserve_bytes=workspace,result_reserve_bytes=buffers,
                     required_peak_bytes=resident+workspace+buffers)
