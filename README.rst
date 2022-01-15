
Commonly Encountered Physics-related Analysis Tasks
===================================================

WORK IN PROGRESS (USE AT YOUR OWN RISK)

Starting up a new project, I often find myself re-using (and more
often than not, re-writing) basic analysis code that I've used 
plenty of times before. This repository is supposed to serve as a
collection of basic functions and/or extended tasks that I have
found useful, and should hopefully be broadly applicable.

This project will almost certainly always be a work in progress,
and will make shameless use of old code, as well as the magical
solutions often found on Stack Overflow.


Install
-------

From sources
````````````

To install system-wide, noting the path to the src since no wheels
exist on PyPI, use::

   pip install ./physics_util

If you intend to edit the code and want the import calls to reflect
those changes, install in developer mode::

   pip install -e physics_util

If you don't want a global installation (i.e. if multiple users will
engage with and/or edit this library) and you don't want to use venv
or some equivalent::

   pip install --user -e physics_util

where pip is pip3 for Python3 (tested on Python 3.6.9). Be careful 
NOT to use ``sudo``, as the latter two installations make a file
``easy-install.pth`` in either the global or the local directory
``lib/python3.X/site-packages/easy-install.pth``, and sudo will
mess up the permissions of this file such that uninstalling is very
complicated.


Uninstall
---------

If installed without ``sudo`` as instructed, uninstalling should be 
as easy as::

   pip uninstall physics_util

If installed using ``sudo`` and with the ``-e`` and ``--user`` flags, 
the above uninstall will encounter an error.

Navigate to the file ``lib/python3.X/site-packages/easy-install.pth``, 
located either at  ``/usr/local/`` or ``~/.local`` and ensure there
is no entry for ``physics_util``.


License
-------

The package is distributed under an open license (see LICENSE file for
information).


Authors
-------

Charles Blakemore (chas.blakemore@gmail.com)