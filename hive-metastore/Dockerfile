FROM apache/hive:3.1.3

USER root

# Must match the Hadoop client jars already inside this image (Hive 3.1.x images usually ship Hadoop 3.1.0).
ARG HADOOP_AWS_VER=3.1.0
# Version aligned with Hadoop 3.1.0 distribution (hadoop-aws POM).
ARG AWS_SDK_BUNDLE_VER=1.11.375

ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VER}/hadoop-aws-${HADOOP_AWS_VER}.jar /tmp/hadoop-aws.jar
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_BUNDLE_VER}/aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VER}.jar /tmp/aws-java-sdk-bundle.jar

RUN set -eux \
 && found=0 \
 && for d in /opt/hive /opt/apache-hive-3.1.3-bin; do \
      if [ -d "${d}/lib" ]; then \
        found=1; \
        install -m0644 /tmp/hadoop-aws.jar "${d}/lib/hadoop-aws.jar"; \
        install -m0644 /tmp/aws-java-sdk-bundle.jar "${d}/lib/aws-java-sdk-bundle.jar"; \
      fi; \
    done \
 && test "${found}" -eq 1 \
 && rm -f /tmp/hadoop-aws.jar /tmp/aws-java-sdk-bundle.jar

USER hive
