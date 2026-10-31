"""Local SSH receive tuning; no remote training code/configuration changes."""
import paramiko


def install():
    original_get=paramiko.SFTPClient.get
    def bounded_get(self,remotepath,localpath,callback=None,prefetch=True,max_concurrent_prefetch_requests=None):
        return original_get(self,remotepath,localpath,callback=callback,prefetch=prefetch,
            max_concurrent_prefetch_requests=64 if max_concurrent_prefetch_requests is None else max_concurrent_prefetch_requests)
    def open_sftp(self):
        return paramiko.SFTPClient.from_transport(self.get_transport(),
            window_size=64*1024**2,max_packet_size=64*1024)
    paramiko.SFTPClient.get=bounded_get
    paramiko.SSHClient.open_sftp=open_sftp
