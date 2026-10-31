# -*- coding: utf-8 -*-
"""Check HRDecoder segmentor class hierarchy + hrdecoder_fcn_hr48 test_cfg."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=25, banner_timeout=25, auth_timeout=25)

    def sh(cmd, timeout=60):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return o.read().decode('utf-8', 'replace').strip()

    print('## HRDecoder.py class def + method overrides:')
    print(sh("grep -nE \"^class |^    def |register_module|LesionEncoderDecoder|"
             "EncoderDecoder\" /root/autodl-tmp/hrplus/repo/mmseg/models/segmentors/"
             "HRDecoder.py | head -30"))
    print('=' * 70)
    print('## hrdecoder_fcn_hr48.py test_cfg / model type:')
    print(sh("grep -nE \"type=|use_sigmoid|my_metrics|compute_aupr|test_cfg\" "
             "/root/autodl-tmp/hrplus/repo/configs/_base_/models/"
             "hrdecoder_fcn_hr48.py | head -20"))
    print('=' * 70)
    print('## HR100Refinement class def (hr100_refinement.py):')
    print(sh("grep -nE \"^class |^    def |LesionEncoderDecoder|EncoderDecoder\" "
             "/root/autodl-tmp/hrplus/repo/hr100_refinement.py | head -30"))
    cli.close()


if __name__ == '__main__':
    main()
